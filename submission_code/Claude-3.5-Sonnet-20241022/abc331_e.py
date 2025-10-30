N, M, L = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Store forbidden combinations in a set for O(1) lookup
forbidden = set()
for _ in range(L):
    c, d = map(int, input().split())
    forbidden.add((c-1, d-1))  # Convert to 0-based indexing

# Find max price among all valid combinations
max_price = 0
for i in range(N):
    for j in range(M):
        if (i, j) not in forbidden:
            price = A[i] + B[j]
            max_price = max(max_price, price)

print(max_price)