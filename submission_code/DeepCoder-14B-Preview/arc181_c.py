import sys
sys.setrecursionlimit(1 << 25)

class SegmentTreeNode:
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.left = None
        self.right = None
        self.sorted_list = []

def build(l, r, arr):
    node = SegmentTreeNode(l, r)
    if l == r:
        node.sorted_list = [arr[l]]
    else:
        mid = (l + r) // 2
        node.left = build(l, mid, arr)
        node.right = build(mid + 1, r, arr)
        # Merge the two sorted lists
        i = j = 0
        merged = []
        while i < len(node.left.sorted_list) and j < len(node.right.sorted_list):
            if node.left.sorted_list[i] < node.right.sorted_list[j]:
                merged.append(node.left.sorted_list[i])
                i += 1
            else:
                merged.append(node.right.sorted_list[j])
                j += 1
        while i < len(node.left.sorted_list):
            merged.append(node.left.sorted_list[i])
            i += 1
        while j < len(node.right.sorted_list):
            merged.append(node.right.sorted_list[j])
            j += 1
        node.sorted_list = merged
    return node

def query_range(node, l, r, x):
    if node.r < l or node.l > r:
        return 0
    if l <= node.l and node.r <= r:
        # Binary search for x in node.sorted_list
        left = 0
        right = len(node.sorted_list)
        while left < right:
            mid = (left + right) // 2
            if node.sorted_list[mid] < x:
                left = mid + 1
            else:
                right = mid
        return left
    return query_range(node.left, l, r, x) + query_range(node.right, l, r, x)

def update_pos(node, index, value):
    if node.l == node.r == index:
        node.sorted_list = [value]
    else:
        if index <= node.left.r:
            update_pos(node.left, index, value)
        else:
            update_pos(node.right, index, value)
        # Merge the children's sorted lists
        i = j = 0
        merged = []
        while i < len(node.left.sorted_list) and j < len(node.right.sorted_list):
            if node.left.sorted_list[i] < node.right.sorted_list[j]:
                merged.append(node.left.sorted_list[i])
                i += 1
            else:
                merged.append(node.right.sorted_list[j])
                j += 1
        while i < len(node.left.sorted_list):
            merged.append(node.left.sorted_list[i])
            i += 1
        while j < len(node.right.sorted_list):
            merged.append(node.right.sorted_list[j])
            j += 1
        node.sorted_list = merged

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, index, delta=1):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    P = list(map(int, data[ptr:ptr+N]))
    ptr += N
    M = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+M]))
    
    # Compute initial inversion count using Fenwick Tree
    max_val = N
    ft = FenwickTree(max_val)
    inversion_count = 0
    for i in reversed(range(N)):
        inversion_count += ft.query(P[i] - 1)
        ft.update(P[i])
    
    # Build the segment tree
    if N == 0:
        root = None
    else:
        root = build(0, N-1, P)
    
    # Process each operation
    for a in A:
        for i in range(a-1):  # 0-based, i from 0 to a-2
            if i >= N-1:
                continue
            if P[i] > P[i+1]:
                x = P[i]
                y = P[i+1]
                
                # Compute C: number of elements < x in [i+2, N-1]
                if i + 2 <= N - 1:
                    C = query_range(root, i + 2, N - 1, x)
                else:
                    C = 0
                
                # Compute D: number of elements < y in [i+1, N-1]
                if i + 1 <= N - 1:
                    D = query_range(root, i + 1, N - 1, y)
                else:
                    D = 0
                
                delta = -1 + C - D
                inversion_count += delta
                
                # Swap the elements
                P[i], P[i+1] = P[i+1], P[i]
                
                # Update the segment tree
                if root is not None:
                    update_pos(root, i, P[i])
                    update_pos(root, i+1, P[i+1])
        
        print(inversion_count)

if __name__ == '__main__':
    main()