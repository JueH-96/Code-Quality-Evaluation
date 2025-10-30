import sys
sys.setrecursionlimit(300000)

class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.data = [-10**18] * (2 * self.size)
    
    def update(self, index, value):
        index += self.size
        if value > self.data[index]:
            self.data[index] = value
        index //= 2
        while index:
            self.data[index] = max(self.data[2*index], self.data[2*index+1])
            index //= 2
            
    def query(self, l, r):
        l += self.size
        r += self.size
        res = -10**18
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.data[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.data[r])
                r -= 1
            l //= 2
            r //= 2
        return res

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        points = []
        all_bs = set()
        for i in range(n):
            a, b = nums1[i], nums2[i]
            s_val = a + b
            points.append((a, b, s_val))
            all_bs.add(b)
        
        queries_indexed = []
        for idx, q in enumerate(queries):
            x, y = q
            queries_indexed.append((x, y, idx))
            all_bs.add(y)
        
        comp = sorted(all_bs)
        comp_map = {val: i for i, val in enumerate(comp)}
        size_comp = len(comp)
        
        events = []
        for a, b, s_val in points:
            events.append((a, b, s_val, 0))
        for x, y, idx in queries_indexed:
            events.append((x, y, idx, 1))
            
        events.sort(key=lambda e: (-e[0], e[3]))
        
        seg_tree = SegmentTree(size_comp)
        ans = [-1] * len(queries)
        
        for event in events:
            if event[3] == 0:
                a_val, b_val, s_val = event[0], event[1], event[2]
                pos = comp_map[b_val]
                seg_tree.update(pos, s_val)
            else:
                x_val, y_val, orig_idx = event[0], event[1], event[2]
                pos_y = comp_map[y_val]
                res = seg_tree.query(pos_y, size_comp-1)
                if res != -10**18:
                    ans[orig_idx] = res
        return ans