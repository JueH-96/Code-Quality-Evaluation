class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, i, delta):
        while i <= self.size:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i):
        result = 0
        while i > 0:
            result += self.tree[i]
            i -= i & (-i)
        return result

def count_inversions_fenwick(arr):
    if not arr:
        return 0
    
    n = len(arr)
    max_val = max(arr)
    ft = FenwickTree(max_val + 1)
    inversions = 0
    
    for i in range(n-1, -1, -1):
        # Query how many elements to the right are smaller
        if arr[i] > 0:
            inversions += ft.query(arr[i])
        # Add current element (with +1 offset since Fenwick tree is 1-indexed)
        ft.update(arr[i] + 1, 1)
    
    return inversions

n, m = map(int, input().split())
a = list(map(int, input().split()))

for k in range(m):
    b = [(a[i] + k) % m for i in range(n)]
    inversions = count_inversions_fenwick(b)
    print(inversions)