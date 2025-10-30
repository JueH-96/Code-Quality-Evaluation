class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i):
        result = 0
        while i > 0:
            result += self.tree[i]
            i -= i & (-i)
        return result

def count_inversions_fast(arr):
    n = len(arr)
    ft = FenwickTree(n)
    inversions = 0
    
    for i in range(n):
        # Count elements to the left with value > arr[i]
        inversions += i - ft.query(arr[i])
        ft.update(arr[i], 1)
    
    return inversions

def apply_operation(P, k):
    swaps = 0
    for i in range(k - 1):
        if P[i] > P[i + 1]:
            P[i], P[i + 1] = P[i + 1], P[i]
            swaps += 1
    return swaps

N = int(input())
P = list(map(int, input().split()))
M = int(input())
A = list(map(int, input().split()))

inv_count = count_inversions_fast(P)

for i in range(M):
    swaps = apply_operation(P, A[i])
    inv_count -= swaps
    print(inv_count)