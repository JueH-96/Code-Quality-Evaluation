import math
import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def sieve(max_n):
            is_prime = [True] * (max_n + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(math.sqrt(max_n)) + 1):
                if is_prime[i]:
                    for j in range(i * i, max_n + 1, i):
                        is_prime[j] = False
            primes = set()
            for num, prime in enumerate(is_prime):
                if prime:
                    primes.add(num)
            return primes
        
        primes = sieve(10**4)
        
        # Initial checks
        if n in primes or m in primes:
            return -1
        
        d = len(str(n))
        min_val = 10 ** (d - 1)
        max_val = (10 ** d) - 1
        
        if d != len(str(m)):
            return -1  # As per problem statement, this should not happen
        
        if n == m:
            return n
        
        max_number = max_val
        INF = float('inf')
        distance = [INF] * (max_number + 1)
        distance[n] = n
        
        heap = []
        heapq.heappush(heap, (n, n))
        
        def get_neighbors(x):
            neighbors = []
            s = list(map(int, str(x)))
            for i in range(d):
                original = s[i]
                for delta in (-1, 1):
                    new_digit = original + delta
                    if i == 0:
                        if new_digit < 1 or new_digit > 9:
                            continue
                    else:
                        if new_digit < 0 or new_digit > 9:
                            continue
                    new_s = s.copy()
                    new_s[i] = new_digit
                    new_num = int(''.join(map(str, new_s)))
                    if min_val <= new_num <= max_val and new_num not in primes:
                        neighbors.append(new_num)
            return neighbors
        
        while heap:
            current_cost, u = heapq.heappop(heap)
            if u == m:
                return current_cost
            if current_cost > distance[u]:
                continue
            for v in get_neighbors(u):
                new_cost = current_cost + v
                if new_cost < distance[v]:
                    distance[v] = new_cost
                    heapq.heappush(heap, (new_cost, v))
        
        return -1