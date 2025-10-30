import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        N = 10000
        is_prime = [True] * (N + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(N ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, N + 1, i):
                    is_prime[j] = False
        
        if is_prime[n] or is_prime[m]:
            return -1
        
        INF = 10**18
        dist = [INF] * (N + 1)
        dist[n] = n
        heap = [(n, n)]
        
        while heap:
            cost, x = heapq.heappop(heap)
            if x == m:
                return cost
            if cost != dist[x]:
                continue
            s = str(x)
            L = len(s)
            for i in range(L):
                digit = int(s[i])
                if digit != 9:
                    new_digit = digit + 1
                    new_s = s[:i] + str(new_digit) + s[i+1:]
                    y = int(new_s)
                    if y <= N and not is_prime[y]:
                        new_cost = cost + y
                        if new_cost < dist[y]:
                            dist[y] = new_cost
                            heapq.heappush(heap, (new_cost, y))
                if digit != 0:
                    new_digit = digit - 1
                    new_s = s[:i] + str(new_digit) + s[i+1:]
                    y = int(new_s)
                    if y <= N and not is_prime[y]:
                        new_cost = cost + y
                        if new_cost < dist[y]:
                            dist[y] = new_cost
                            heapq.heappush(heap, (new_cost, y))
        return -1