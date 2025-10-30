import heapq
from math import isqrt

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # Determine the number of digits to work with (we preserve leading zeros)
        d = len(str(n))
        max_limit = 10 ** d  # All numbers with d digits: 0 to 10^d - 1
        
        # Build a sieve of primes in the range [0, max_limit)
        sieve = [True] * max_limit
        if max_limit > 0:
            sieve[0] = False
        if max_limit > 1:
            sieve[1] = False
        for i in range(2, isqrt(max_limit) + 1):
            if sieve[i]:
                for j in range(i * i, max_limit, i):
                    sieve[j] = False

        # If the starting number or target number is prime, transformation is impossible.
        if sieve[n] or sieve[m]:
            return -1

        # Use Dijkstra's algorithm.
        # Each state is a number with d digits (using zfill to maintain leading zeros).
        # The cost of a path is the sum of the numbers in the sequence.
        INF = float('inf')
        dist = [INF] * max_limit
        dist[n] = n  # start cost includes the starting state's value.
        
        # Heap will hold (total_cost, state) pairs.
        heap = [(n, n)]
        
        while heap:
            cur_cost, cur_state = heapq.heappop(heap)
            if cur_cost != dist[cur_state]:
                continue
            if cur_state == m:
                return cur_cost
            
            # Represent state as a fixed-length string.
            s = str(cur_state).zfill(d)
            for i in range(d):
                digit = int(s[i])
                
                # Operation 1: Increase this digit by 1 if it's not '9'
                if digit < 9:
                    new_digit = digit + 1
                    new_state_str = s[:i] + str(new_digit) + s[i+1:]
                    new_state = int(new_state_str)
                    # Only proceed if the new state is not prime.
                    if new_state < max_limit and not sieve[new_state]:
                        new_cost = cur_cost + new_state
                        if new_cost < dist[new_state]:
                            dist[new_state] = new_cost
                            heapq.heappush(heap, (new_cost, new_state))
                
                # Operation 2: Decrease this digit by 1 if it's not '0'
                if digit > 0:
                    new_digit = digit - 1
                    new_state_str = s[:i] + str(new_digit) + s[i+1:]
                    new_state = int(new_state_str)
                    if new_state < max_limit and not sieve[new_state]:
                        new_cost = cur_cost + new_state
                        if new_cost < dist[new_state]:
                            dist[new_state] = new_cost
                            heapq.heappush(heap, (new_cost, new_state))
        
        # If we exit the loop without reaching m, it's not possible.
        return -1