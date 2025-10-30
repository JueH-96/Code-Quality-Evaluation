class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # --------------------------------------------
        # 1) Quick checks and setup
        # --------------------------------------------
        # If n or m is prime, it's impossible right away
        # because n must never be prime, and the final m must not be prime.
        if self.is_prime(n) or self.is_prime(m):
            return -1
        
        # If n == m, the cost is just n (the sum of values visited is only n itself).
        if n == m:
            return n
        
        # We will perform a Dijkstra-like search (D') over all 0..9999 states.
        # Each state is an integer of the same digit-length as n/m.
        # The "cost" of moving from x to y is simply "y", because
        # the total cost of a path is the sum of all states visited.
        
        # We'll keep track of the best cost to reach each state to avoid reprocessing.
        
        from heapq import heappush, heappop
        import math
        
        # Precompute primes up to 9999 so prime checks are O(1).
        # (Alternatively, we could have used self.is_prime each time, but a sieve is faster.)
        
        digit_len = len(str(n))
        
        # best_cost[x] will hold the minimum sum of visited states when reaching state x.
        best_cost = [math.inf] * 10000
        best_cost[n] = n
        
        # Min-heap for (cost_so_far, state)
        heap = [(n, n)]
        
        while heap:
            current_cost, current_state = heappop(heap)
            
            # If we've already found a cheaper cost for this state, skip
            if current_cost > best_cost[current_state]:
                continue
            
            # If we reach m, we're done
            if current_state == m:
                return current_cost
            
            # Enumerate neighbors by incrementing/decrementing each digit if allowed
            neighbors = self.next_states(current_state, digit_len)
            
            for nxt in neighbors:
                # Cost to get to nxt = current_cost + nxt
                new_cost = current_cost + nxt
                if new_cost < best_cost[nxt]:
                    best_cost[nxt] = new_cost
                    heappush(heap, (new_cost, nxt))
        
        # If we exhaust the search without reaching m, it's impossible
        return -1
    
    def next_states(self, x: int, digit_len: int):
        """
        Generate all valid neighbors of x by increasing or decreasing
        any digit (if not 9 or 0, respectively), skipping primes.
        """
        s = str(x).zfill(digit_len)
        for i in range(digit_len):
            digit = int(s[i])
            # Try +1
            if digit < 9:
                new_digit = digit + 1
                new_num = int(s[:i] + str(new_digit) + s[i+1:])
                if not self.is_prime(new_num):
                    yield new_num
            # Try -1
            if digit > 0:
                new_digit = digit - 1
                new_num = int(s[:i] + str(new_digit) + s[i+1:])
                if not self.is_prime(new_num):
                    yield new_num
    
    def is_prime(self, x: int) -> bool:
        """
        Check primality for an integer x < 10^4 (x >= 0).
        """
        if x < 2:
            return False
        if x < 4:
            return True  # 2 and 3 are prime
        if x % 2 == 0 or x % 3 == 0:
            return x == 2 or x == 3
        r = int(x**0.5)
        for i in range(5, r+1, 6):
            if x % i == 0 or x % (i+2) == 0:
                return False
        return True