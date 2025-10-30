from collections import Counter

n, q = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

for _ in range(q):
    l, r, L, R = map(int, input().split())
    
    # Count frequencies in the ranges (convert to 0-based indexing)
    count_A = Counter(A[l-1:r])
    count_B = Counter(B[L-1:R])
    
    # Check if they match
    if count_A == count_B:
        print("Yes")
    else:
        print("No")