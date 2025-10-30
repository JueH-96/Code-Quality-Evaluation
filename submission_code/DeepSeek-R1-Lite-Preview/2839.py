from typing import List

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        q = len(queries)
        
        # Step 1: Coordinate compression for nums2
        sorted_nums2 = sorted(set(nums2))
        rank = {num: idx + 1 for idx, num in enumerate(sorted_nums2)}  # 1-based indexing

        # Step 2: Sort points by nums1 descending
        points = sorted(zip(nums1, nums2), key=lambda x: -x[0])
        sums = [points[i][0] + points[i][1] for i in range(n)]
        
        # Step 3: Sort queries by x_i descending, keeping original indices
        sorted_queries = sorted([(x, y, idx) for idx, (x, y) in enumerate(queries)], key=lambda x: -x[0])
        
        # Step 4: Build a segment tree for range maximum queries on nums2's ranks
        size = len(sorted_nums2) + 2  # To handle 1-based indexing
        seg = [float('-inf')] * (2 * size)
        
        def update(pos, value):
            pos += size  # 1-based indexing
            if seg[pos] < value:
                seg[pos] = value
                pos //= 2
                while pos >= 1:
                    seg[pos] = max(seg[2*pos], seg[2*pos+1])
                    pos //= 2
        
        def query(l, r):
            l += size
            r += size
            res = float('-inf')
            while l <= r:
                if l % 2 == 1:
                    res = max(res, seg[l])
                    l += 1
                if r % 2 == 0:
                    res = max(res, seg[r])
                    r -= 1
                l //= 2
                r //= 2
            return res if res != float('-inf') else -1
        
        # Step 5: Process sorted queries
        answer = [0] * q
        point_ptr = 0  # Pointer to add points to the segment tree
        for x, y, idx in sorted_queries:
            # Add all points with nums1[j] >= x
            while point_ptr < n and points[point_ptr][0] >= x:
                r = rank[points[point_ptr][1]]
                s = points[point_ptr][0] + points[point_ptr][1]
                update(r, s)
                point_ptr += 1
            # Find the rank of y
            y_rank = rank.get(y, -1)
            if y_rank == -1:
                # y is smaller than any nums2[j], so query entire range
                y_rank = 1
            else:
                # Find the smallest rank where nums2[j] >= y
                y_rank = next(i for i, num in enumerate(sorted_nums2) if num >= y) + 1
            # Query from y_rank to the maximum rank
            res = query(y_rank, len(sorted_nums2))
            answer[idx] = res
        return answer