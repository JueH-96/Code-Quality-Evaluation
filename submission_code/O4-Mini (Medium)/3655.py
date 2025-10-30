import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # If start or end is prime, impossible
        # n and m must never be prime
        # cost is sum of all node values visited (including start and end)
        # moves: increment or decrement any one digit by 1,
        # keeping same number of digits (no leading zero if multi-digit),
        # and resultant number must not be prime
        
        # Quick prime sieve up to 10000
        MAXV = 10000
        is_prime = [True] * (MAXV + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(MAXV**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, MAXV+1, i):
                    is_prime[j] = False
        
        # If either n or m is prime, no valid path
        if is_prime[n] or is_prime[m]:
            return -1
        
        # If already equal, cost is just n
        if n == m:
            return n
        
        s_n = str(n)
        D = len(s_n)
        # For D>1, forbid leading zeros; for D==1, allow 0..9
        low = 0 if D == 1 else 10**(D-1)
        high = 10**D - 1
        
        # Dijkstra: dist[x] = min cost to reach x (including sum of node values)
        INF = 10**18
        dist = [INF] * (high + 1)
        dist[n] = n
        heap = [(n, n)]  # (cost, node)
        
        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist[u]:
                continue
            if u == m:
                return cost
            s = str(u).zfill(D)
            # generate neighbors
            for i in range(D):
                d = int(s[i])
                for delta in (-1, 1):
                    nd = d + delta
                    if nd < 0 or nd > 9:
                        continue
                    # no leading zero if multi-digit
                    if i == 0 and nd == 0 and D > 1:
                        continue
                    new_s = s[:i] + str(nd) + s[i+1:]
                    v = int(new_s)
                    # stay in digit-length range
                    if v < low or v > high:
                        continue
                    # must not be prime
                    if is_prime[v]:
                        continue
                    new_cost = cost + v
                    if new_cost < dist[v]:
                        dist[v] = new_cost
                        heapq.heappush(heap, (new_cost, v))
        
        return -1