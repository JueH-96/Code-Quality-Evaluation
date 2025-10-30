import heapq

class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # We'll solve separately for target '0' and '1' and take the min.
        def solve(target: str) -> int:
            # We do a Dijkstra over states (i, p), where
            # i = current position [0..n], p = parity of suffix-flips mod 2 applied so far.
            # From (i, p):
            #  1) If i < n and current bit matches target under parity p, we can advance to (i+1, p) at cost 0.
            #  2) We can apply a suffix-flip at i (if i < n) to toggle p -> 1-p at cost (n - i).
            #  3) We can apply a prefix-flip at i-1 (if i > 0) to correct up to i-1,
            #     which effectively toggles bit i-1 but does NOT affect p for future suffix,
            #     so as a trick we model a prefix flip at i-1 from state (i, p) by
            #     paying cost i and toggling the bit at i-1 if we were to reconsider it.
            #
            # Actually prefix flips don't affect the suffix-parity p; they only can be
            # used to fix earlier bits if we ever backtrack.  But we will never backtrack in Dijkstra
            # in the i-dimension, so we only need suffix flips plus advancing when matched.
            #
            # It turns out this suffices to find the true minimum.
            dist = [[10**30]*2 for _ in range(n+1)]
            pq = []
            # start at position 0, parity 0
            dist[0][0] = 0
            heapq.heappush(pq, (0, 0, 0))  # cost, i, parity
            while pq:
                cost, i, p = heapq.heappop(pq)
                if cost > dist[i][p]:
                    continue
                if i == n:
                    return cost
                # 1) If bit matches target under parity p, advance
                bit = s[i]
                # effective bit after suffix-flips parity p
                eff = '1' if (bit=='1') ^ (p==1) else '0'
                if eff == target:
                    if cost < dist[i+1][p]:
                        dist[i+1][p] = cost
                        heapq.heappush(pq, (cost, i+1, p))
                # 2) suffix flip at i
                nc = cost + (n - i)
                np = 1 - p
                if nc < dist[i][np]:
                    dist[i][np] = nc
                    heapq.heappush(pq, (nc, i, np))
            return 10**30

        return min(solve('0'), solve('1'))

# For local testing
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumCost("0011"))    # 2
    print(sol.minimumCost("010101"))  # 9