import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # Convert to fixed-length strings
        s_n = str(n)
        s_m = str(m)
        k = len(s_n)
        # Quick checks: start and target must not be prime
        max_val = 10 ** k
        # Sieve primes up to max_val (covering 0..10^k-1)
        sieve = [True] * (max_val)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_val**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, max_val, i):
                    sieve[j] = False
        
        if sieve[n] or sieve[m]:
            return -1
        
        # Dijkstra: distance[state_str] = min cost to reach state (sum of ints)
        INF = 10**18
        dist = {}
        start = s_n
        target = s_m
        start_cost = n
        dist[start] = start_cost
        pq = [(start_cost, start)]
        
        while pq:
            cost, s = heapq.heappop(pq)
            if cost > dist[s]:
                continue
            if s == target:
                return cost
            # generate neighbors
            for i in range(k):
                d = ord(s[i]) - ord('0')
                # increase digit
                if d < 9:
                    t = s[:i] + chr(ord('0') + d + 1) + s[i+1:]
                    v = int(t)
                    if not sieve[v]:
                        new_cost = cost + v
                        if new_cost < dist.get(t, INF):
                            dist[t] = new_cost
                            heapq.heappush(pq, (new_cost, t))
                # decrease digit
                if d > 0:
                    t = s[:i] + chr(ord('0') + d - 1) + s[i+1:]
                    v = int(t)
                    if not sieve[v]:
                        new_cost = cost + v
                        if new_cost < dist.get(t, INF):
                            dist[t] = new_cost
                            heapq.heappush(pq, (new_cost, t))
        
        return -1