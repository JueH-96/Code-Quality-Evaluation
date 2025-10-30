import sys
import bisect

def main():
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    H = list(map(int, sys.stdin.readline().split()))
    
    stack = []
    L = [0] * (N + 1)
    for j in range(1, N + 1):
        while stack and H[stack[-1] - 1] <= H[j - 1]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        else:
            L[j] = 0
        stack.append(j)
    
    a = [0] * (N + 1)
    for j in range(1, N + 1):
        a[j] = L[j] + 1
    
    class SegmentTree:
        class Node:
            def __init__(self, start, end):
                self.start = start
                self.end = end
                self.left = None
                self.right = None
                self.data = []
        
        def __init__(self, a):
            self.n = len(a) - 1
            self.root = self.build_tree(1, self.n, a)
        
        def build_tree(self, start, end, a):
            node = self.Node(start, end)
            if start == end:
                node.data = [a[start]]
            else:
                mid = (start + end) // 2
                node.left = self.build_tree(start, mid, a)
                node.right = self.build_tree(mid + 1, end, a)
                node.data = node.left.data + node.right.data
                node.data.sort()
            return node
        
        def query_range(self, node, l, r, target):
            if node.end < l or node.start > r:
                return 0
            if l <= node.start and node.end <= r:
                return bisect.bisect_right(node.data, target)
            else:
                return self.query_range(node.left, l, r, target) + self.query_range(node.right, l, r, target)
    
    st = SegmentTree(a)
    
    for _ in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        j_start = max(r + 1, l + 1)
        if j_start > N:
            print(0)
        else:
            count = st.query_range(st.root, j_start, N, l)
            print(count)

if __name__ == "__main__":
    main()