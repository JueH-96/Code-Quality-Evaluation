# YOUR CODE HERE
N, M, L = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

forbidden = set()
for _ in range(L):
    c, d = map(int, input().split())
    forbidden.add((c-1, d-1))  # Convert to 0-indexed

max_price = 0
for i in range(N):
    for j in range(M):
        if (i, j) not in forbidden:
            max_price = max(max_price, a[i] + b[j])

print(max_price)