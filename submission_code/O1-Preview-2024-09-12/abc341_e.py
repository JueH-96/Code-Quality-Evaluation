# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    S_input = sys.stdin.readline().strip()
    S_orig = [int(c) for c in S_input]
    N = len(S_orig)
    
    class SegmentTreeSNode:
        def __init__(self, l, r):
            self.l = l
            self.r = r
            self.left = None
            self.right = None
            self.flip = False
            self.val = None  # Only used for leaf nodes
            if l == r:
                self.val = S_orig[l - 1]
            else:
                m = (l + r) // 2
                self.left = SegmentTreeSNode(l, m)
                self.right = SegmentTreeSNode(m +1 , r)
        
        def push(self):
            if self.flip and self.left:
                self.left.flip ^= True
                self.right.flip ^= True
                self.flip = False

        def range_flip(self, L, R):
            if self.r < L or self.l > R:
                return
            if L <= self.l and self.r <= R:
                self.flip ^= True
            else:
                self.push()
                self.left.range_flip(L, R)
                self.right.range_flip(L, R)

        def point_query(self, idx):
            if self.l == self.r:
                return self.val ^ self.flip
            self.push()
            if idx <= self.left.r:
                return self.left.point_query(idx)
            else:
                return self.right.point_query(idx)
                
    class SegmentTreeAdjNode:
        def __init__(self, l, r, adj_equal):
            self.l = l
            self.r = r
            self.left = None
            self.right = None
            self.sum = 0
            if l == r:
                self.sum = adj_equal[l - 1]
            else:
                m = (l + r) // 2
                self.left = SegmentTreeAdjNode(l, m, adj_equal)
                self.right = SegmentTreeAdjNode(m + 1, r, adj_equal)
                self.sum = self.left.sum + self.right.sum

        def range_sum(self, L, R):
            if self.r < L or self.l > R:
                return 0
            if L <= self.l and self.r <= R:
                return self.sum
            return self.left.range_sum(L, R) + self.right.range_sum(L, R)

        def point_update(self, idx, val):
            if self.l == self.r:
                self.sum = val
            else:
                if idx <= self.left.r:
                    self.left.point_update(idx, val)
                else:
                    self.right.point_update(idx, val)
                self.sum = self.left.sum + self.right.sum
    
    adj_equal = []
    for i in range(N-1):
        adj_equal.append(int(S_orig[i] == S_orig[i+1]))
    if N > 1:
        adj_tree = SegmentTreeAdjNode(1, N-1, adj_equal)
    else:
        adj_tree = None
    S_tree = SegmentTreeSNode(1, N)
    
    output = []
    q_lines = [sys.stdin.readline().strip() for _ in range(Q)]
    for q_line in q_lines:
        parts = q_line.strip().split()
        if parts[0] == '1':
            # Flip operation
            _, L_str, R_str = parts
            L = int(L_str)
            R = int(R_str)
            S_tree.range_flip(L, R)
            # Update adj_equal[L-1], adj_equal[R] if needed
            if N > 1:
                if L > 1:
                    v1 = S_tree.point_query(L - 1)
                    v2 = S_tree.point_query(L)
                    adj_equal_L_minus_1 = int(v1 == v2)
                    adj_tree.point_update(L -1, adj_equal_L_minus_1)
                if R < N:
                    v1 = S_tree.point_query(R)
                    v2 = S_tree.point_query(R + 1)
                    adj_equal_R = int(v1 == v2)
                    adj_tree.point_update(R, adj_equal_R)
        elif parts[0] == '2':
            # Query operation
            _, L_str, R_str = parts
            L = int(L_str)
            R = int(R_str)
            if N == 1 or L == R:
                output.append('Yes')
            else:
                res = adj_tree.range_sum(L, R -1 )
                if res == 0:
                    output.append('Yes')
                else:
                    output.append('No')
    for ans in output:
        print(ans)
threading.Thread(target=main).start()