# YOUR CODE HERE

N, M, L = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

bad_pairs = set()
for _ in range(L):
    c, d = map(int, input().split())
    bad_pairs.add((c-1, d-1))

max_price = 0
for i in range(N):
    for j in range(M):
        if (i, j) not in bad_pairs:
            max_price = max(max_price, a[i] + b[j])

print(max_price)