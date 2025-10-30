class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        NEG_INF = -10**18
        
        # Segment tree where each node stores:
        # dp[i][j] = max sum where i indicates if we take left endpoint, j indicates if we take right endpoint
        tree = [[[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]] for _ in range(4 * n)]
        
        def build(node, l, r):
            if l == r:
                tree[node][0][0] = 0  # don't take the element
                tree[node][1][1] = nums[l]  # take the element
                # dp[0][1] and dp[1][0] are invalid for single element
                tree[node][0][1] = NEG_INF
                tree[node][1][0] = NEG_INF
            else:
                mid = (l + r) // 2
                build(2 * node, l, mid)
                build(2 * node + 1, mid + 1, r)
                merge(node, 2 * node, 2 * node + 1)
        
        def merge(parent, left, right):
            lf = tree[left]
            rf = tree[right]
            f = tree[parent]
            
            # When merging, we can't take both right endpoint of left segment 
            # and left endpoint of right segment (they're adjacent)
            f[0][0] = max(lf[0][0] + rf[0][0], lf[0][1] + rf[0][0], lf[0][0] + rf[1][0])
            f[0][1] = max(lf[0][0] + rf[0][1], lf[0][1] + rf[0][1], lf[0][0] + rf[1][1])
            f[1][0] = max(lf[1][0] + rf[0][0], lf[1][1] + rf[0][0], lf[1][0] + rf[1][0])
            f[1][1] = max(lf[1][0] + rf[0][1], lf[1][1] + rf[0][1], lf[1][0] + rf[1][1])
        
        def update(node, l, r, pos, val):
            if l == r:
                tree[node][0][0] = 0
                tree[node][1][1] = val
                tree[node][0][1] = NEG_INF
                tree[node][1][0] = NEG_INF
            else:
                mid = (l + r) // 2
                if pos <= mid:
                    update(2 * node, l, mid, pos, val)
                else:
                    update(2 * node + 1, mid + 1, r, pos, val)
                merge(node, 2 * node, 2 * node + 1)
        
        build(1, 0, n - 1)
        
        total = 0
        for pos, x in queries:
            update(1, 0, n - 1, pos, x)
            f = tree[1]
            # Take maximum of all valid states, ensuring non-negative
            max_sum = max(f[0][0], f[0][1], f[1][0], f[1][1], 0)
            total = (total + max_sum) % MOD
        
        return total