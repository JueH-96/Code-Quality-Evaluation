class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # Helper function: build sieve up to N (inclusive),
        # returns a list is_prime where is_prime[x] is True if x is prime.
        def build_sieve(N):
            sieve = [True] * (N+1)
            sieve[0] = sieve[1] = False
            p = 2
            while p * p <= N:
                if sieve[p]:
                    for multiple in range(p*p, N+1, p):
                        sieve[multiple] = False
                p += 1
            return sieve
        
        # Determine the fixed digit count based on n's length (n and m have same digit count)
        dig = len(str(n))
        max_val = 10**dig - 1
        
        # Build sieve for numbers in range [0, max_val]. 
        # Note: 0 and 1 are not prime.
        is_prime = build_sieve(max_val)
        
        # If starting or target number is prime, transformation is impossible.
        if is_prime[n] or is_prime[m]:
            return -1

        # Use Dijkstra’s algorithm.
        # Each state is an integer that, when represented as a fixed-width string
        # of length 'dig', its digits can be manipulated.
        # The cost is defined as the sum of the values of n at each visited state,
        # including the initial state.
        import heapq
        INF = float('inf')
        dist = [INF] * (10**dig)
        dist[n] = n  # start cost includes n itself
        
        # Priority queue: (current total cost, integer state)
        heap = [(n, n)]
        
        while heap:
            cost, x = heapq.heappop(heap)
            if cost != dist[x]:
                continue
            if x == m:
                return cost
            
            # Represent x as fixed-width string.
            sx = f"{x:0{dig}d}"
            # Try all possible operations: increase or decrease one digit.
            for i in range(dig):
                digit = int(sx[i])
                # For each position, two possible moves: if digit < 9 then +1; if digit > 0 then -1.
                for delta in (1, -1):
                    new_digit = digit + delta
                    if 0 <= new_digit <= 9:
                        # Create new state string: replace character at position i.
                        ns = sx[:i] + str(new_digit) + sx[i+1:]
                        nx = int(ns)
                        # The next state's integer value must not be prime.
                        if is_prime[nx]:
                            continue
                        new_cost = cost + nx
                        if new_cost < dist[nx]:
                            dist[nx] = new_cost
                            heapq.heappush(heap, (new_cost, nx))
        
        return -1
                        
# You can test the solution with the sample examples:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.minOperations(10, 12))  # Expected output: 85
    # Example 2
    print(sol.minOperations(4, 8))    # Expected output: -1
    # Example 3
    print(sol.minOperations(6, 2))    # Expected output: -1