class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        a = [nums[i] - i for i in range(n)]
        
        # Coordinate compression
        sorted_a = sorted(set(a))
        coord = {v: i for i, v in enumerate(sorted_a)}
        m = len(sorted_a)
        
        # Segment tree
        tree = [float('-inf')] * (4 * m)
        
        def update(node, start, end, idx, val):
            if start == end:
                tree[node] = max(tree[node], val)
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    update(2 * node, start, mid, idx, val)
                else:
                    update(2 * node + 1, mid + 1, end, idx, val)
                tree[node] = max(tree[2 * node], tree[2 * node + 1])
        
        def query(node, start, end, l, r):
            if r < start or l > end:
                return float('-inf')
            if l <= start and end <= r:
                return tree[node]
            mid = (start + end) // 2
            left_max = query(2 * node, start, mid, l, r)
            right_max = query(2 * node + 1, mid + 1, end, l, r)
            return max(left_max, right_max)
        
        result = float('-inf')
        for i in range(n):
            compressed = coord[a[i]]
            max_prev = query(1, 0, m - 1, 0, compressed)
            dp_i = max(nums[i], max_prev + nums[i])
            update(1, 0, m - 1, compressed, dp_i)
            result = max(result, dp_i)
        
        return result