"""Composable metric reducers and streaming trajectory logger.

O(neurons) activity summaries per sampled tick; arbitrary-large run files are
optional. Event logs store all (state,action,next_state,reward,done,collision)
needed to reconstruct empirical occupancy/behavior, not internal neuron rates.
"""
from abc import ABC,abstractmethod
from pathlib import Path
import json,csv
import numpy as np

class Metric(ABC):
    @abstractmethod
    def observe(self,**kwargs): ...
    @abstractmethod
    def result(self): ...

class OnlineNeuronActivity(Metric):
    """Welford per-neuron mean/variance; report average firing and silent fraction.

    use one separate instance per recorded time period (e.g. phase or episode).
    """
    def __init__(self,n):self.n=0;self.mean=np.zeros(n);self.m2=np.zeros(n);self.active=np.zeros(n,dtype=np.int64)
    def observe(self,**kwargs):
        rates=np.asarray(kwargs['rates'],dtype=float)
        if rates.shape!=self.mean.shape:raise ValueError('rate dimensionality changed')
        self.n+=1;diff=rates-self.mean;self.mean+=diff/self.n
        self.m2+=diff*(rates-self.mean)
        self.active+=(rates>1.e-9)
    def result(self):
        return {'n_samples':self.n,'mean':self.mean.tolist(),
                'variance':(self.m2/max(1,self.n-1)).tolist(),
                'active_fraction':(self.active/max(1,self.n)).tolist()}

class OccupancyMatrix(Metric):
    """Empirical state transition counts, not analytical SR or oracle P."""
    def __init__(self,n):self.counts=np.zeros((n,n),dtype=np.int64);self.action_counts=np.zeros((n,4),dtype=np.int64)
    def observe(self,**kwargs):
        s=int(kwargs['state']);ns=int(kwargs['next_state']);a=int(kwargs['action'])
        self.counts[s,ns]+=1;self.action_counts[s,a]+=1
    def result(self):
        rows=self.counts.sum(axis=1,keepdims=True)
        probabilities=np.divide(self.counts,rows,out=np.zeros_like(self.counts,dtype=float),where=rows>0)
        return {'counts':self.counts,'transition_probabilities':probabilities,'action_counts':self.action_counts}

class EpisodeMetrics(Metric):
    def __init__(self):self.reset()
    def reset(self):self.steps=0;self.reward=0.;self.collisions=0;self.success=False
    def observe(self,**kwargs):
        self.steps+=1;self.reward+=float(kwargs['reward']);self.collisions+=int(kwargs.get('collision',False));self.success|=bool(kwargs.get('success',False))
    def result(self):return {'reward':self.reward,'steps':self.steps,'success':int(self.success),'collisions':self.collisions}

class MetricsLogger:
    def __init__(self,root,n_states,n_ca1,save_events=True,save_rates=False,rate_stride=1,max_rate_buffer=1024):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.save_events=save_events;self.save_rates=save_rates;self.rate_stride=rate_stride
        self.max_rate_buffer=max_rate_buffer
        if max_rate_buffer<1:raise ValueError('max_rate_buffer >=1')
        if rate_stride<1:raise ValueError('rate_stride >=1')
        self.occupancy=OccupancyMatrix(n_states);self.activity=OnlineNeuronActivity(n_ca1)
        self.episode=EpisodeMetrics();self.summary=[];self._rates=[];self._rate_ticks=[]
        self.phase_activity={};self.current_phase=None;self.rate_shards=[]
        self._events=(self.root/'events.jsonl').open('w') if save_events else None
        self._observations=(self.root/'observations.jsonl').open('w') if save_events else None
        self.current_episode=None
    def begin_phase(self,phase):
        if self.current_phase is not None:raise RuntimeError('phase already active')
        self.current_phase=str(phase)
        self.phase_activity.setdefault(self.current_phase,OnlineNeuronActivity(len(self.activity.mean)))
    def end_phase(self):
        self.current_phase=None
    def begin_episode(self,episode):self.current_episode=int(episode)
    def presentation(self,state,start_tick,n_ticks):
        if self._observations:
            self._observations.write(json.dumps({'episode':self.current_episode,'phase':self.current_phase,
                'state':int(state),'tick_start':int(start_tick),'n_ticks':int(n_ticks)})+'\n')
    def _flush_rate_buffer(self):
        if not self._rates:return
        path=self.root/f'rates_part_{len(self.rate_shards):05d}.npz'
        np.savez_compressed(path,rates=np.asarray(self._rates,dtype=np.float32),
                            ticks=np.asarray(self._rate_ticks,dtype=np.int64))
        self.rate_shards.append(path.name)
        self._rates.clear();self._rate_ticks.clear()
    def tick(self,rate,tick):
        self.activity.observe(rates=rate)
        if self.current_phase is not None:self.phase_activity[self.current_phase].observe(rates=rate)
        if self.save_rates and tick%self.rate_stride==0:
            self._rates.append(np.asarray(rate,dtype=np.float32).copy());self._rate_ticks.append(tick)
            if len(self._rates)>=self.max_rate_buffer:self._flush_rate_buffer()
    def transition(self,episode,phase,state,action,next_state,reward,terminated,truncated,collision,success):
        item={'episode':int(episode),'phase':str(phase),'state':int(state),'action':int(action),
              'next_state':int(next_state),'reward':float(reward),'terminated':bool(terminated),
              'truncated':bool(truncated),'collision':bool(collision),'success':bool(success)}
        self.episode.observe(reward=reward,collision=collision,success=success)
        self.occupancy.observe(state=state,action=action,next_state=next_state)
        if self._events:self._events.write(json.dumps(item)+'\n')
    def finish_episode(self,episode,phase,start,spl,success_override=None):
        if success_override is not None:
            self.episode.success=bool(success_override)
        item={'episode':int(episode),'phase':str(phase),'start':int(start),**self.episode.result(),'spl':float(spl)}
        self.summary.append(item);self.episode.reset()
    def finish(self):
        if self._events:self._events.close();self._events=None
        if self._observations:self._observations.close();self._observations=None
        if self.summary:
            with (self.root/'episodes.csv').open('w',newline='') as f:
                out=csv.DictWriter(f,fieldnames=list(self.summary[0]));out.writeheader();out.writerows(self.summary)
        with (self.root/'activity.json').open('w') as f:json.dump(self.activity.result(),f)
        (self.root/'activity_by_phase.json').write_text(json.dumps({k:v.result() for k,v in self.phase_activity.items()}))
        np.savez_compressed(self.root/'occupancy.npz',**self.occupancy.result())
        if self.save_rates:
            if self.rate_shards:
                self._flush_rate_buffer()
            else:
                np.savez_compressed(self.root/'rates.npz',rates=np.asarray(self._rates,dtype=np.float32),ticks=np.asarray(self._rate_ticks))
                self.rate_shards.append('rates.npz')
            (self.root/'rates_index.json').write_text(json.dumps({'shards':self.rate_shards,'stride':self.rate_stride}))
        return self.summary
