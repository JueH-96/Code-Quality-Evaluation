import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        k = len(str(n))
        max_state = 10**k - 1
        
        if max_state == 0:
            is_prime = [False]
        else:
            is_prime = [True] * (max_state + 1)
            is_prime[0] = False
            if max_state >= 1:
                is_prime[1] = False
            for i in range(2, int(max_state**0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, max_state + 1, i):
                        is_prime[j] = False
        
        if n > max_state or m > max_state or is_prime[n] or is_prime[m]:
            return -1
        
        INF = 10**18
        dist = [INF] * (max_state + 1)
        dist[n] = n
        heap = [(n, n)]
        
        while heap:
            cost, state = heapq.heappop(heap)
            if state == m:
                return cost
            if cost != dist[state]:
                continue
            for place in range(k):
                divisor = 10**place
                digit = (state // divisor) % 10
                if digit != 9:
                    new_digit = digit + 1
                    new_state = state - digit * divisor + new_digit * divisor
                    if 0 <= new_state <= max_state and not is_prime[new_state]:
                        new_cost = cost + new_state
                        if new_cost < dist[new_state]:
                            dist[new_state] = new_cost
                            heapq.heappush(heap, (new_cost, new_state))
                if digit != 0:
                    new_digit = digit - 1
                    new_state = state - digit * divisor + new_digit * divisor
                    if 0 <= new_state <= max_state and not is_prime[new_state]:
                        new_cost = cost + new_state
                        if new_cost < dist[new_state]:
                            dist[new_state] = new_cost
                            heapq.heappush(heap, (new_cost, new_state))
        return -1