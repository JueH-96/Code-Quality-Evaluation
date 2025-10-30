# Read the input
N, M, L = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
bad_pairs = set()
for _ in range(L):
    c, d = map(int, input().split())
    bad_pairs.add((c, d))

# Find the most expensive set meal
max_price = 0
for i in range(1, N+1):
    for j in range(1, M+1):
        if (i, j) not in bad_pairs:
            price = a[i-1] + b[j-1]
            max_price = max(max_price, price)

# Print the answer
print(max_price)