class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        from bisect import bisect_left, bisect_right
        import bisect
        n = len(nums1)
        m = len(queries)
        all_nums2 = set()
        for val in nums2:
            all_nums2.add(val)
        for x_i, y_i in queries:
            all_nums2.add(y_i)
        sorted_nums2 = sorted(all_nums2)
        nums2_index = {val: idx for idx, val in enumerate(sorted_nums2)}
        num_indices = len(sorted_nums2)
        # Initialize segment tree
        class SegmentTree:
            def __init__(self, size):
                self.N = 1
                while self.N < size:
                    self.N <<=1
                self.size = size
                self.tree = [-1]*(2*self.N)
            def update(self, idx, val):
                idx += self.N
                self.tree[idx] = max(self.tree[idx], val)
                while idx > 1:
                    idx >>=1
                    self.tree[idx] = max(self.tree[2*idx], self.tree[2*idx+1])
            def query(self, l, r):
                l += self.N
                r += self.N
                res = -1
                while l < r:
                    if l%2:
                        res = max(res, self.tree[l])
                        l +=1
                    if r%2:
                        r -=1
                        res = max(res, self.tree[r])
                    l >>=1
                    r >>=1
                return res
        st = SegmentTree(num_indices)
        events = []
        for i in range(n):
            events.append( (-nums1[i], -nums2[i], nums1[i]+nums2[i], 'data') )
        for idx, (x_i, y_i) in enumerate(queries):
            events.append( (-x_i, -y_i, idx, 'query') )
        events.sort()
        ans = [-1]*len(queries)
        for event in events:
            if event[3] == 'data':
                nums1_i = -event[0]
                nums2_i = -event[1]
                sum_i = event[2]
                idx = nums2_index[nums2_i]
                st.update(idx, sum_i)
            else:
                x_i = -event[0]
                y_i = -event[1]
                idx_query = event[2]
                idx = bisect_left(sorted_nums2, y_i)
                res = st.query(idx, num_indices)
                ans[idx_query] = res
        return ans