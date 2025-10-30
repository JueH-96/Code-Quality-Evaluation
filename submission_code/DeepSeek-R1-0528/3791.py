class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        seg = [-1] * (4 * n)
        
        def build(node, l, r):
            if l == r:
                seg[node] = baskets[l]
            else:
                mid = (l + r) // 2
                build(2 * node + 1, l, mid)
                build(2 * node + 2, mid + 1, r)
                seg[node] = max(seg[2 * node + 1], seg[2 * node + 2])
        
        def update(node, l, r, idx):
            if l == r:
                seg[node] = -1
            else:
                mid = (l + r) // 2
                if idx <= mid:
                    update(2 * node + 1, l, mid, idx)
                else:
                    update(2 * node + 2, mid + 1, r, idx)
                seg[node] = max(seg[2 * node + 1], seg[2 * node + 2])
        
        def query(node, l, r, x):
            if seg[node] < x:
                return -1
            if l == r:
                return l
            mid = (l + r) // 2
            if seg[2 * node + 1] >= x:
                return query(2 * node + 1, l, mid, x)
            else:
                return query(2 * node + 2, mid + 1, r, x)
        
        build(0, 0, n - 1)
        unplaced = 0
        for f in fruits:
            idx = query(0, 0, n - 1, f)
            if idx == -1:
                unplaced += 1
            else:
                update(0, 0, n - 1, idx)
        return unplaced