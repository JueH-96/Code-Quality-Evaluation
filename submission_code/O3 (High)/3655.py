import heapq
import math

class Solution:
    # ------------------------------------------------------------
    # Helper : test primality (for the early impossible checks)
    # ------------------------------------------------------------
    @staticmethod
    def _is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        r = int(math.isqrt(x))
        for d in range(3, r + 1, 2):
            if x % d == 0:
                return False
        return True

    # ------------------------------------------------------------
    # Helper : sieve of Eratosthenes up to 'limit'
    # ------------------------------------------------------------
    @staticmethod
    def _sieve(limit: int):
        prime = [True] * (limit + 1)
        if limit >= 0:
            prime[0] = False
        if limit >= 1:
            prime[1] = False
        p = 2
        while p * p <= limit:
            if prime[p]:
                for multiple in range(p * p, limit + 1, p):
                    prime[multiple] = False
            p += 1
        return prime

    # ------------------------------------------------------------
    # Main routine
    # ------------------------------------------------------------
    def minOperations(self, n: int, m: int) -> int:
        # They must have the same number of digits according
        # to the statement (not re-checked here)

        # If either start or target is prime, transformation
        # is impossible (they have to be composite/non-prime all along)
        if self._is_prime(n) or self._is_prime(m):
            return -1

        # Trivial case : already equal (and both non-prime)
        if n == m:
            return n                      # cost is just the initial value

        # Number of digits determines the search space
        d = len(str(n))
        limit = 10 ** d                  # numbers 0 … 10^d − 1  (leading zeros allowed)

        # Pre-compute primality for every possible state
        is_prime = self._sieve(limit - 1)
        if is_prime[n] or is_prime[m]:   # re-check with the sieve (saves us from primes>9999 if ever)
            return -1

        # Pre-compute the place values 1,10,100,… for quick neighbour generation
        place_values = [10 ** i for i in range(d)]

        INF = float('inf')
        dist = [INF] * limit             # minimal cost seen so far for every value
        dist[n] = n                      # cost includes the starting value itself
        pq = [(n, n)]                    # (current_cost, current_value)

        while pq:
            cost, val = heapq.heappop(pq)
            if cost != dist[val]:        # stale entry
                continue

            if val == m:                 # reached the target
                return cost

            # ----------------------------------------------------
            # Generate neighbours by ±1 in each digit
            # ----------------------------------------------------
            for p in place_values:
                digit = (val // p) % 10

                # Increase this digit (if it is not 9)
                if digit < 9:
                    nxt = val + p
                    if not is_prime[nxt]:
                        new_cost = cost + nxt
                        if new_cost < dist[nxt]:
                            dist[nxt] = new_cost
                            heapq.heappush(pq, (new_cost, nxt))

                # Decrease this digit (if it is not 0)
                if digit > 0:
                    nxt = val - p
                    if not is_prime[nxt]:
                        new_cost = cost + nxt
                        if new_cost < dist[nxt]:
                            dist[nxt] = new_cost
                            heapq.heappush(pq, (new_cost, nxt))

        # If we emptied the queue without reaching 'm'
        return -1