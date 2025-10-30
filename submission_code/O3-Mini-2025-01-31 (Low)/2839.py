class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # We'll use an offline approach with a Binary Indexed Tree (Fenwick Tree) for range maximum queries.
        # Steps:
        # 1. Preprocess: For each index j from nums1/nums2, we want to later update BIT at coordinate (nums2[j]).
        # 2. For each query [x, y], we want the maximum sum for indices with nums1[j] >= x and nums2[j] >= y.
        #    We sort queries in descending order by x.
        # 3. Also, sort the indices by nums1[j] in descending order.
        # 4. We'll use coordinate compression on the nums2 values (and query y values) to allow BIT indexing.
        #
        # BIT: Will support update and query for maximum in a range.
        #
        # Complexity: O((n + q) * log(n + q))
        
        n = len(nums1)
        q = len(queries)
        
        # Build coordinate list for nums2 values and query y values.
        coords = []
        for num in nums2:
            coords.append(num)
        for _x, y in queries:
            coords.append(y)
        # remove duplicates and sort
        coords = sorted(set(coords))
        # mapping from value to coordinate index (1-indexed for BIT)
        comp = {v: i + 1 for i, v in enumerate(coords)}
        size = len(coords)
        
        # BIT implementation for maximum queries.
        class BIT:
            def __init__(self, n):
                self.n = n
                self.tree = [ -10**18 ] * (n + 1)  # initialize with very negative values
            
            def update(self, idx, value):
                while idx <= self.n:
                    if value > self.tree[idx]:
                        self.tree[idx] = value
                    idx += idx & -idx
            
            def query(self, idx):
                # returns maximum on [1, idx]
                res = -10**18
                while idx > 0:
                    if self.tree[idx] > res:
                        res = self.tree[idx]
                    idx -= idx & -idx
                return res
        
        # We want to query for nums2[j] >= y. 
        # But BIT query as implemented gives maximum for prefix [1, idx].
        # To answer "nums2[j] >= y" after compression, note:
        # Let cy = comp[y]. Then all indices with compressed value in [cy, size] will have real value >= y.
        # So, if we maintain BIT in reverse order (i.e., update reversed index: r = size - comp + 1) 
        # OR we can do a query on BIT that gives maximum on suffix [cy, size].
        #
        # A trick: Instead of rewriting BIT for suffix queries, we can 
        # update BIT at index "size - comp[nums2[j]] + 1" and then for query y, query BIT on prefix [size - comp[y] + 1].
        
        # Let's transform:
        # For each j, new index = rev_index = size - comp[nums2[j]] + 1.
        # For query with y, we want all j's with nums2[j] >= y -> comp[nums2[j]] >= comp[y] 
        # <=> rev_index = size - comp[nums2[j]] + 1 <= size - comp[y] + 1.
        # So we want to query BIT for indices in range [1, size - comp[y] + 1].
        
        bit = BIT(size)
        
        # Sort indices by nums1 descending
        idxs = list(range(n))
        idxs.sort(key=lambda j: nums1[j], reverse=True)
        
        # Prepare queries with index and sorted descending by x (here x = query[0] as given)
        q_list = []
        for i, (x, y) in enumerate(queries):
            q_list.append((x, y, i))
        q_list.sort(key=lambda t: t[0], reverse=True)
        
        ans = [-1] * q
        
        j = 0
        # Process queries in order of descending x.
        for x, y, qi in q_list:
            # While there are indices with nums1[j] >= x, update BIT.
            while j < n and nums1[idxs[j]] >= x:
                cur_index = idxs[j]
                # get BIT index corresponding to nums2[cur_index]
                rev_index = size - comp[nums2[cur_index]] + 1
                # value to update is (nums1 + nums2)
                current_val = nums1[cur_index] + nums2[cur_index]
                bit.update(rev_index, current_val)
                j += 1
            # For the query, compute BIT query on prefix up to: R = size - comp[y] + 1 if y exists.
            # Check if y is in comp... Since we added queries to comp, it is.
            target = size - comp[y] + 1
            best = bit.query(target)
            if best <= -10**18//2:  # means no valid update
                ans[qi] = -1
            else:
                ans[qi] = best
        return ans