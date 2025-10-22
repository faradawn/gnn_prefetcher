from collections import deque
class DequeLRU():
    def __init__(self, maxsize):
        self.cache = deque()
        self.marks = {}
        # left MRU right LRU
        self.maxsize = maxsize
        self.total_ios = 0
        self.total_hits = 0
        self.total_pres = 0
        self.total_prehits = 0

    def _boost(self, lba):
        self.cache.remove(lba)
        self.cache.appendleft(lba)

    def __contains__(self, lba):
        return lba in self.cache

    def __repr__(self):
        return str(self.cache)

    def get_size(self):
        return len(self.cache)

    def get_last(self):
        return self.cache[-1]

    def get_first(self):
        return self.cache[0]

    def remove(self, lba):
        self.cache.remove(lba)
        return True

    def frict(self):
        return self.cache.popleft()

    def evict(self):
        return self.cache.pop()

    def full(self):
        return len(self.cache) == self.maxsize

    def push_back(self, lba):
        self.cache.append(lba)
        assert len(self.cache) <= self.maxsize
    # push the new, return the popped lba else None

    # check if lba is in cache
    def check(self, lba):
        self.total_ios += 1
        if lba in self.cache:
            self.total_prehits += 1
            self.total_hits += 1
            return True
        else:
            return False
    
    def push(self, lba, lbamark='p'):
        self.total_pres += 1
        if lba in self.cache:
            self._boost(lba)
        else:
            if self.full():
                self.cache.pop()
            self.cache.appendleft(lba)

    def get_hit_rate(self):
        return self.total_hits / (self.total_ios + 1e-16)

    def get_prehit_rate(self):
        return self.total_prehits / (self.total_ios + 1e-16)

    def get_stats(self):
        return self.total_ios,self.total_pres,self.total_hits,self.total_prehits


class CacheTest():

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.cache = DequeLRU(maxsize=maxsize)

    def __contains__(self, lba):
        return lba in self.cache

    def __repr__(self):
        return self.cache.__repr__()

    #push normal io
    def push_normal(self,lba):
        return self.cache.push(lba=lba,lbamark='n')

    #push prefetch io
    def push_prefetch(self,lba,lazy_prefetch = False):
        if lazy_prefetch:
            if lba in self.cache:
                return None
            else:
                return self.cache.push(lba=lba,lbamark='p')
        else:
            return self.cache.push(lba=lba,lbamark='p')
    def check(self, lba):
        return self.cache.check(lba)
    
    def get_hit_rate(self):
        return  self.cache.get_hit_rate()

    def get_prehit_rate(self):
        return  self.cache.get_prehit_rate()

    def get_stats(self):
        return  self.cache.get_stats()