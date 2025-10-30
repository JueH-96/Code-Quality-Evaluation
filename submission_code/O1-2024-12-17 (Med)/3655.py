class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # ------------------------------------------------------------
        # 1) Quick checks: if n or m is prime, return -1 immediately.
        #    Also handle case n == m (if n == m and not prime, cost = n).
        # ------------------------------------------------------------
        
        # Precompute primes up to 9999 via sieve
        MAX_VAL = 9999
        is_prime = [True]*(MAX_VAL+1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(MAX_VAL**0.5)+1):
            if is_prime[i]:
                for j in range(i*i, MAX_VAL+1, i):
                    is_prime[j] = False
        
        # Check if n or m is prime
        if is_prime[n] or is_prime[m]:
            return -1
        
        # If n == m, the cost is just n (since we don't need to do any operations)
        if n == m:
            return n  # n not prime is ensured above
        
        # ------------------------------------------------------------
        # 2) We will use Dijkstra's algorithm (or uniform cost search)
        #    to find the minimum sum of states from n to m.
        #    dist[x] will track the minimal sum of visited numbers
        #    to reach x (including x itself in the sum).
        # ------------------------------------------------------------
        
        from heapq import heappush, heappop
        
        # Number of digits to ensure we stay in the same "length"
        digit_len = len(str(n))
        
        def same_length(x: int) -> bool:
            # Check if x has the same number of digits as n
            return len(str(x)) == digit_len
        
        # Function to generate neighbors by changing one digit up or down
        def neighbors(x: int):
            s = list(str(x))
            for i in range(len(s)):
                d = int(s[i])
                # Increment if possible
                if d < 9:
                    s[i] = str(d + 1)
                    new_x = int("".join(s))
                    if same_length(new_x):
                        yield new_x
                # revert
                s[i] = str(d)
                
                # Decrement if possible
                if d > 0:
                    s[i] = str(d - 1)
                    new_x = int("".join(s))
                    if same_length(new_x):
                        yield new_x
                # revert
                s[i] = str(d)
        
        # Dijkstra's algorithm
        INF = float('inf')
        dist = [INF]*(MAX_VAL+1)
        dist[n] = n  # cost includes the starting integer
        heap = [(n, n)]  # (cost, number)
        
        while heap:
            curr_cost, x = heappop(heap)
            if x == m:
                return curr_cost  # Found the best cost to m
            if curr_cost > dist[x]:
                continue
            
            for nxt in neighbors(x):
                if not is_prime[nxt]:  # must never be prime
                    new_cost = curr_cost + nxt
                    if new_cost < dist[nxt]:
                        dist[nxt] = new_cost
                        heappush(heap, (new_cost, nxt))
        
        # If we exhaust the search without reaching m, it's impossible
        return -1