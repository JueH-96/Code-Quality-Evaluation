# YOUR CODE HERE
N, M, L = map(int, input().split())
main_dishes = list(map(int, input().split()))
side_dishes = list(map(int, input().split()))

# Create a set of forbidden combinations
forbidden = set()
for _ in range(L):
    c, d = map(int, input().split())
    forbidden.add((c-1, d-1))  # Convert to 0-based indexing

max_price = 0

# Find the maximum price
for i in range(N):
    for j in range(M):
        if (i, j) not in forbidden:
            price = main_dishes[i] + side_dishes[j]
            max_price = max(max_price, price)

print(max_price)