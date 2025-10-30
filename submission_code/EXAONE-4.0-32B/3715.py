from collections import defaultdict

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        segs = coins
        n = len(segs)
        candidates = set()
        events = []
        
        for i, (L, R, c) in enumerate(segs):
            candidates.add(L)
            candidates.add(R + 1)
            candidates.add(L - k + 1)
            candidates.add(R - k + 1)
            
            events.append((L - k + 1, 'activate', i))
            events.append((R - k + 1, 'to_group1', i))
            events.append((L, 'to_group1a', i))
            events.append((R + 1, 'deactivate', i))
        
        event_dict = defaultdict(list)
        for ev in events:
            x, typ, i = ev
            event_dict[x].append((typ, i))
        
        candidate_xs = sorted(candidates)
        
        state = [None] * n
        s1a = s1b = s2a = s2b = 0
        t1a = t1b = t2a = t2b = 0
        
        ans = 0
        
        for x in candidate_xs:
            if x in event_dict:
                events_here = event_dict[x]
                events_here.sort(key=lambda e: {'activate': 0, 'to_group1': 1, 'to_group1a': 2, 'deactivate': 3}[e[0]])
                
                for typ, i in events_here:
                    L, R, c = segs[i]
                    if typ == 'activate':
                        if L <= x:
                            s2a += c
                            state[i] = 'group2a'
                        else:
                            s2b += c
                            t2b += (k - L) * c
                            state[i] = 'group2b'
                    elif typ == 'to_group1':
                        if state[i] == 'group2a':
                            s2a -= c
                        elif state[i] == 'group2b':
                            s2b -= c
                            t2b -= (k - L) * c
                        if L <= x:
                            s1a += c
                            t1a += (R + 1) * c
                            state[i] = 'group1a'
                        else:
                            s1b += (R - L + 1) * c
                            state[i] = 'group1b'
                    elif typ == 'to_group1a':
                        if state[i] == 'group1b':
                            s1b -= (R - L + 1) * c
                            s1a += c
                            t1a += (R + 1) * c
                            state[i] = 'group1a'
                    elif typ == 'deactivate':
                        if state[i] is None:
                            continue
                        if state[i] == 'group1a':
                            s1a -= c
                            t1a -= (R + 1) * c
                        elif state[i] == 'group1b':
                            s1b -= (R - L + 1) * c
                        elif state[i] == 'group2a':
                            s2a -= c
                        elif state[i] == 'group2b':
                            s2b -= c
                            t2b -= (k - L) * c
                        state[i] = None
            
            total = x * (-s1a + s2b) + (t1a + s1b + k * s2a + t2b)
            if total > ans:
                ans = total
                
        return ans