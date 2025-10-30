import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        # Helper: sieve to mark primes up to 9999
        MAXV = 10000
        is_prime = [False, False] + [True] * (MAXV - 2)
        for i in range(2, int(MAXV**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, MAXV, i):
                    is_prime[j] = False

        # If start or end is prime, impossible
        if is_prime[n] or is_prime[m]:
            return -1
        # If already equal
        if n == m:
            return n

        # Determine digit length and bounds
        s_n = str(n)
        d = len(s_n)
        low = 10**(d-1)
        high = 10**d - 1

        # Dijkstra: dist[x] = min total cost to reach x (including sum of all nodes visited)
        INF = 10**18
        dist = [INF] * MAXV
        dist[n] = n  # cost includes the starting value
        pq = [(n, n)]  # (current total cost, node)

        while pq:
            cost, x = heapq.heappop(pq)
            if cost > dist[x]:
                continue
            if x == m:
                return cost

            # Generate neighbors by changing one digit ±1
            sx = list(str(x))
            for i in range(d):
                orig_digit = int(sx[i])
                for delta in (-1, 1):
                    nd = orig_digit + delta
                    # skip invalid digit changes
                    if nd < 0 or nd > 9:
                        continue
                    # no leading zero
                    if i == 0 and nd == 0:
                        continue
                    sx[i] = str(nd)
                    y = int("".join(sx))
                    sx[i] = str(orig_digit)

                    # skip primes
                    if is_prime[y]:
                        continue

                    new_cost = cost + y
                    if new_cost < dist[y]:
                        dist[y] = new_cost
                        heapq.heappush(pq, (new_cost, y))

        # unreachable
        return -1