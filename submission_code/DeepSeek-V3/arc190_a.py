import sys

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr + 2*N]))
        ptr += 2*N
        
        pos = [[] for _ in range(N+1)]
        for idx, num in enumerate(A):
            pos[num].append(idx)
        
        non_adj = []
        for num in range(1, N+1):
            p1, p2 = pos[num]
            if abs(p1 - p2) != 1:
                non_adj.append(num)
        
        non_adj.sort()
        m = len(non_adj)
        res = 0
        
        # For each a and b in non_adj, check if their positions are interleaved
        # The total possible pairs is C(m, 2) minus those pairs that are not interleaved.
        # So we need to count the number of pairs (a, b) where the intervals [a1, a2] and [b1, b2] are interleaved.
        # To compute this efficiently, we can process the intervals and use a Fenwick tree.
        
        # First, collect all intervals (l, r) where l and r are the positions of the numbers in non_adj, sorted such that l < r.
        intervals = []
        for num in non_adj:
            p1, p2 = pos[num]
            if p1 > p2:
                p1, p2 = p2, p1
            intervals.append((p1, p2))
        
        # Sort intervals by their right endpoint in ascending order
        intervals.sort(key=lambda x: x[1])
        
        # Assign ranks to the left endpoints for coordinate compression
        sorted_lefts = sorted([x[0] for x in intervals])
        # We can use bisect to find the rank of any left endpoint
        
        from bisect import bisect_right
        
        # Fenwick tree to handle prefix sums
        class FenwickTree:
            def __init__(self, size):
                self.size = size
                self.tree = [0] * (self.size + 2)
            
            def update(self, index, delta=1):
                while index <= self.size:
                    self.tree[index] += delta
                    index += index & -index
            
            def query(self, index):
                res = 0
                while index > 0:
                    res += self.tree[index]
                    index -= index & -index
                return res
        
        max_rank = len(sorted_lefts)
        ft = FenwickTree(max_rank)
        
        total_pairs = 0
        for i, (l, r) in enumerate(intervals):
            # The number of intervals processed before this one where left > current l
            # The total such intervals is i - ft.query(rank of l)
            rank = bisect_right(sorted_lefts, l)
            cnt = ft.query(rank)
            total_pairs += (i - cnt)
            ft.update(rank)
        
        res = total_pairs
        results.append(res)
    
    print('
'.join(map(str, results)))

solve()