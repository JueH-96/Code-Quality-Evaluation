class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        """
        We want to count the number of subarrays of nums where:
          1) The first and last elements of the subarray are the same,
          2) That element is also the maximum in the subarray.

        Key observation:
          If a subarray has its maximum element = x, and both ends are x,
          then all elements in the subarray are <= x, and the subarray
          starts and ends with x.

        One way to count these efficiently is to process the array in
        ascending order of values and use a Disjoint Set Union (DSU)
        structure to keep track of "active" indices (those whose value
        has been encountered/processed). Indices become active when
        we process their value. Once active, they can be union-ed with
        adjacent active indices if those adjacent indices have values
        <= the current value (which is guaranteed by processing in
        ascending order).

        For each value group v:
          - We activate the indices that have value = v (make them
            singletons in the DSU).
            Each such index immediately contributes 1 new subarray
            (the single-element subarray).
          - We then union each newly activated index with its active
            neighbor(s). Merging two components that contain c1 and c2
            active indices of value v respectively will form c1*c2 new
            subarrays (each choice of one of the c1 indices with one
            of the c2 indices forms a new [start, end] of the same
            value v and no bigger value in between).

        Complexity:
          - We gather all indices by their values, sort these values
            (O(n log n) in the worst case).
          - For each index, we do up to two union operations in DSU
            (amortized α(n) per operation, nearly O(1)*).
          - Hence total is O(n log n) dominated by sorting.

        This correctly counts all subarrays meeting the stated condition.
        """

        import sys
        sys.setrecursionlimit(10**7)

        # 1) Collect indices by their value
        from collections import defaultdict
        val_to_indices = defaultdict(list)
        for i, val in enumerate(nums):
            val_to_indices[val].append(i)

        # 2) Sort the distinct values in ascending order
        unique_vals = sorted(val_to_indices.keys())

        # DSU (Disjoint Set Union) / Union-Find structure
        parent = list(range(len(nums)))
        size = [1] * len(nums)  # for rank/size union
        # We'll also keep track of how many indices of the "current value"
        # are in each component. This count resets for the new value.
        # We'll store it in a separate array that we re-initialize
        # whenever we move to a new value.
        count_curr_val = [0] * len(nums)

        # "active" marks if an index is in the DSU yet (i.e., has been processed)
        active = [False] * len(nums)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            rootA = find(a)
            rootB = find(b)
            if rootA != rootB:
                # union by size/rank
                if size[rootA] < size[rootB]:
                    rootA, rootB = rootB, rootA
                # attach rootB under rootA
                parent[rootB] = rootA
                size[rootA] += size[rootB]
                # c1, c2 = how many of this value in each root
                c1 = count_curr_val[rootA]
                c2 = count_curr_val[rootB]
                # merging gives c1*c2 new subarrays
                res_add = c1 * c2
                # update the DSU count for rootA
                count_curr_val[rootA] = c1 + c2
                return res_add
            return 0

        ans = 0

        # 3) Process values in ascending order
        for v in unique_vals:
            indices_list = val_to_indices[v]
            # First, make each new index i active as a singleton
            for i in indices_list:
                active[i] = True
                parent[i] = i
                size[i] = 1
                count_curr_val[i] = 1  # exactly one index of value v here
                # Single-element subarray
                ans += 1

            # Now try to union each with its active neighbor
            for i in indices_list:
                # check neighbors
                if i - 1 >= 0 and active[i - 1]:
                    ans += union(i, i - 1)
                if i + 1 < len(nums) and active[i + 1]:
                    ans += union(i, i + 1)

        return ans