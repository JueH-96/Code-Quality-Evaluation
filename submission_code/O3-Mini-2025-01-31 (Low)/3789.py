from typing import List
import sys

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Build candidate lists:
        # For each index i=1..n (as a) store all candidate values (b - 1) coming from a conflict pair [i, b]
        # Note: only consider pair if a < b.
        sortedCandidates = [[] for _ in range(n+1)]
        for a, b in conflictingPairs:
            if a > b:
                a, b = b, a
            # conflict pair [a, b] with a < b
            sortedCandidates[a].append(b - 1)
        # sort each candidate list and record the minimal candidate for index i
        g = [n] * (n+1)   # index 1..n
        # for indices with no conflict starting there: g[i] remains n.
        for i in range(1, n+1):
            if sortedCandidates[i]:
                sortedCandidates[i].sort()
                g[i] = sortedCandidates[i][0]
        
        # f[i] = minimum over j from i to n of g[j]
        f = [n] * (n+2)  # f[n+1] = n, index 1..n
        f[n+1] = n
        f[n] = g[n]
        for i in range(n-1, 0, -1):
            f[i] = g[i] if g[i] < f[i+1] else f[i+1]
        
        # Count base valid subarrays:
        base_total = 0
        for l in range(1, n+1):
            if f[l] >= l:
                base_total += (f[l] - l + 1)
        # The answer is at least base_total (if removal doesn't change anything).
        ans = base_total

        # We now simulate removals only for indices where removal makes a difference.
        # For a given index "a", if sortedCandidates[a] has more than one candidate, then by removing the conflict pair that gave g[a] (i.e. sortedCandidates[a][0])
        # we can update g[a] to new value = sortedCandidates[a][1]. Then recompute f' for positions 1..a.
        # Let new_g[i] = same as g[i] for all i != a, and new_g[a] = new_val.
        # Then for i <= a, f_new[i] = min(new_g[i], new_g[i+1], ..., new_g[a], f[a+1]).
        # We'll recompute for i = a downto 1.
        for a in range(1, n+1):
            if not sortedCandidates[a]:
                continue
            # We try the removal only if this conflict removal would change g[a]. 
            # Note: if there is only one candidate in sortedCandidates[a], then removal sets g[a] = n.
            # Also, if sortedCandidates[a][0] is already n then no change would occur.
            old_val = g[a]
            if len(sortedCandidates[a]) >= 2:
                new_val = sortedCandidates[a][1]
            else:
                new_val = n
            if new_val <= old_val:
                # removal not improving (should be new_val >= old_val) – but just in case.
                continue
            # Create a copy of new_f for indices 1..a.
            fnew = [n] * (a+2)
            fnew[a+1] = f[a+1]  # remains unchanged beyond index a.
            # For index a: new candidate = new_val.
            fnew[a] = new_val if new_val < f[a+1] else f[a+1]
            for i in range(a-1, 0, -1):
                # fnew[i] = min(g[i] (unchanged), fnew[i+1])
                fnew[i] = g[i] if g[i] < fnew[i+1] else fnew[i+1]
            # Now compute the contribution from positions 1..a using fnew, and from a+1..n using original f.
            tot = 0
            for i in range(1, a+1):
                if fnew[i] >= i:
                    tot += (fnew[i] - i + 1)
            for i in range(a+1, n+1):
                if f[i] >= i:
                    tot += (f[i] - i + 1)
            if tot > ans:
                ans = tot
        return ans

# For local testing, you can run the following:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    n1 = 4
    conflictingPairs1 = [[2,3],[1,4]]
    print(sol.maxSubarrays(n1, conflictingPairs1))  # Expected output: 9
    # Example 2:
    n2 = 5
    conflictingPairs2 = [[1,2],[2,5],[3,5]]
    print(sol.maxSubarrays(n2, conflictingPairs2))  # Expected output: 12