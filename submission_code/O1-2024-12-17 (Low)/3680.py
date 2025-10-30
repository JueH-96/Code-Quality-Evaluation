class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        """
        We have n distinct integers in nums and a positive integer threshold.
        Build a graph of n nodes (each node corresponds to one element in nums),
        and connect two nodes i, j by an undirected edge if LCM(nums[i], nums[j]) <= threshold.
        We must return the number of connected components of this graph.
        
        Key observations / approach:
        1) If a number x in nums is strictly greater than threshold, then for any other number y >= 1,
           LCM(x, y) >= x > threshold. So that node is isolated (no edges to anything else).
           Hence, all numbers > threshold are each alone in their own connected component.

        2) Only the numbers <= threshold can possibly have edges among themselves. We must find
           how those numbers (call that set S1) connect under the condition LCM(a, b) <= threshold.

        3) A classical way (used in similar problems) to avoid O(n^2) checks is:
           - Focus on integers k = 1..threshold.
           - Among the multiples of k that appear in S1 (i.e., k, 2k, 3k, ... <= threshold),
             we can check consecutive multiples to see if they are connected.
           - In particular, consider k*i and k*(i+1). If both appear in S1 and 
             LCM(k*i, k*(i+1)) <= threshold, then we union them in a DSU (disjoint-set / union-find).

        4) Why consecutive multiples?
           - Let a = k*i, b = k*(i+1). Then gcd(a, b) >= k, and
             LCM(a, b) = (a * b) / gcd(a, b).
             Because they differ by k, gcd(a, b) = k (since i and i+1 are coprime),
             so LCM(a, b) = (k*i) * (k*(i+1)) / k = k * i * (i+1).
           - We only union them if k*i*(i+1) <= threshold.

        5) Implementation details:
           - Partition nums into S2 = { x | x>threshold }, which are all isolated.
           - S1 = { x | x<=threshold }.
           - Store a dictionary val_to_idx so we know which index in nums each S1-value corresponds to.
           - Initialize a union-find (DSU) data structure for the S1 portion.
           - For each k = 1..threshold:
               - Walk multiples m = k, 2k, 3k, ... up to <= threshold.
               - Keep track of the last multiple of k found in S1, call it prev.
               - When we find a new multiple cur in S1, if prev is not None and
                 k * ( (prev//k) ) * ( (prev//k) + 1 ) <= threshold
                 (where (prev//k) and (cur//k) should be consecutive integers),
                 then union the DSU sets of val_to_idx[prev] and val_to_idx[cur].
                 - Then update prev = cur.
           - Count how many distinct DSU parents are in S1.
           - Add len(S2) to that DSU count to get the final result.

        6) Complexity:
           - We iterate k = 1..threshold. For each k, we go through multiples k, 2k, 3k, ...
             The total number of steps is about threshold * (1 + 1/2 + 1/3 + ... + 1/threshold)
             ~ threshold * log(threshold). For threshold up to 2e5, this is on the order of a couple
             million operations, which can be done in optimized Python. We do union-find (amortized ~ α(n))
             which is very small. This should be acceptable if implemented carefully.

        Let's implement the described approach.
        """

        # ----------------
        # 0) Edge case: if threshold == 0 (though constraints say threshold >= 1),
        #    every number > 0 is isolated. But not applicable here. We'll just follow the general logic.
        # ----------------

        import math

        # Separate S1 (<= threshold) and S2 (> threshold).
        # Also build a map from the value in S1 -> index in the original nums array.
        # We only need the union-find for S1. S2 are automatically isolated.
        bigger = 0  # count how many > threshold
        s1_vals = []
        val_to_idx = {}
        
        # We also need the original indices for union-find size = len(S1).
        # But we only care about connectivity among S1, so we'll build DSU of size = number_of_S1.
        
        # Step 1: filter
        n = len(nums)
        for i, v in enumerate(nums):
            if v > threshold:
                bigger += 1
            else:
                s1_vals.append(v)
        
        # If no elements <= threshold, answer = n (all are disconnected).
        if not s1_vals:
            return n
        
        s1_vals.sort()
        # index_in_s1[value] = position in s1_vals (0-based).
        val_to_idx = {}
        for i, v in enumerate(s1_vals):
            val_to_idx[v] = i

        m = len(s1_vals)  # number of values <= threshold

        # DSU (union-find) for the m elements in s1_vals
        parent = list(range(m))
        rank = [0]*m

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                # union by rank
                if rank[pa] < rank[pb]:
                    parent[pa] = pb
                elif rank[pa] > rank[pb]:
                    parent[pb] = pa
                else:
                    parent[pb] = pa
                    rank[pa] += 1

        # Step 2: For each k in [1..threshold], connect consecutive multiples if they remain within threshold
        # and if the LCM <= threshold.  For consecutive multiples k*i and k*(i+1),
        # LCM = k * i * (i+1).
        # We'll do:
        #    multiple = k, 2k, 3k, ... up to <= threshold
        #    keep track of the "previous" multiple we saw in s1. 
        #    If the new multiple is exactly prev + k, then i+1 = (prev//k) + 1.
        #    We can check if k * (   (prev//k) ) * ( (prev//k) + 1 ) <= threshold.
        #    If so, union them.

        # We'll store presence in a set/dict for O(1) membership
        present_s1 = set(s1_vals)

        # For performance, we do an integer i from 1..(threshold // k). 
        # multiple = k*i
        # We'll remember last_multiple that was in s1.
        for k in range(1, threshold+1):
            # As soon as k*k > threshold, sometimes it might be smaller in iteration times, but let's keep it direct.
            # We'll gather multiples up to threshold.
            # This can be done with increment: multiple += k each time.
            multiple = k
            prev_mult = None
            prev_i = None
            i = 1
            while multiple <= threshold:
                if multiple in present_s1:
                    if prev_mult is not None:
                        # check if i == prev_i + 1 (consecutive in terms of i)
                        # if yes, then LCM(k*prev_i, k*i) = k * prev_i * i
                        if i == prev_i + 1:
                            # check if k*prev_i*i <= threshold
                            if k * prev_i * i <= threshold:
                                # union
                                union(
                                    val_to_idx[prev_mult],
                                    val_to_idx[multiple]
                                )
                    prev_mult = multiple
                    prev_i = i
                i += 1
                multiple += k
        
        # Count number of distinct parents in s1
        roots = set(find(x) for x in range(m))
        cnt_s1 = len(roots)

        # Add the components that come from elements > threshold (all isolated)
        return cnt_s1 + bigger