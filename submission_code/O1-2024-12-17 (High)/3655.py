class Solution:
    def minOperations(self, n: int, m: int) -> int:
        """
        We want to transform integer n into integer m by incrementing/decrementing
        digits, under these rules:
         1) Each operation changes exactly one digit by +1 or -1.
         2) We can only increment a digit if it is not 9, and only decrement if it is not 0.
         3) At no point (initially or after any operation) can the integer be prime.
         4) The cost is the sum of all intermediate integer values along the path (including n and m).
         5) Return the minimum possible cost, or -1 if it is impossible.

        Both n and m have the same number of digits, up to 4 digits each (1 <= n,m < 10^4).
        """

        # Step 1: Quick checks if n or m is prime => immediately impossible
        #         Also, if n == m (and not prime), cost is just n.
        if self.is_prime(n) or self.is_prime(m):
            return -1
        if n == m:
            return n  # no operations needed, cost is just n itself

        # Number of digits
        s_n = str(n)
        d = len(s_n)

        # Step 2: We'll run a Dijkstra's algorithm on the space of all d-digit integers,
        #         allowing leading zeros. That is integers from 0..(10^d - 1), all
        #         interpreted as d-digit strings. We'll skip prime integers.

        # Precompute prime status up to 9999 (this covers up to 4 digits).
        # We only need to do this once, but it's fine to embed here for clarity.
        max_val = 10 ** d  # upper bound for d-digit space
        is_prime = self.sieve_primes(9999)

        # Distance array: dist[x] will hold the minimum sum-of-values cost
        # to reach x from n (inclusive of all intermediate states).
        INF = float('inf')
        dist = [INF] * max_val
        dist[n] = n

        import heapq
        pq = [(n, n)]  # (cost_so_far, current_integer)

        while pq:
            cost, current = heapq.heappop(pq)
            if cost > dist[current]:
                continue
            if current == m:
                return cost  # Found a minimal-cost path to m

            # Get all neighbors by changing each digit +1 or -1 if possible
            neighbors = self.get_neighbors(current, d)
            for nxt in neighbors:
                if is_prime[nxt]:  # cannot become prime
                    continue
                new_cost = cost + nxt
                if new_cost < dist[nxt]:
                    dist[nxt] = new_cost
                    heapq.heappush(pq, (new_cost, nxt))

        return -1  # If m was never reached

    def sieve_primes(self, limit: int):
        """Return a list is_prime[0..limit], where is_prime[x] = True if x is prime."""
        is_prime = [True] * (limit + 1)
        if limit >= 0:
            is_prime[0] = False
        if limit >= 1:
            is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return is_prime

    def is_prime(self, x: int) -> bool:
        """Quick check for primality up to 9999 (for small x)."""
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    def get_neighbors(self, x: int, d: int):
        """
        Given an integer x, produce all valid neighbors by +/-1 in one digit,
        within a d-digit representation (leading zeros allowed).
        """
        s = f"{x:0{d}d}"  # d-digit, zero-padded string
        for i in range(d):
            digit = int(s[i])
            # Increment if digit < 9
            if digit < 9:
                new_str = s[:i] + str(digit + 1) + s[i+1:]
                yield int(new_str)
            # Decrement if digit > 0
            if digit > 0:
                new_str = s[:i] + str(digit - 1) + s[i+1:]
                yield int(new_str)