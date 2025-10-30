# YOUR CODE HERE

# Read input
N, M, L = map(int, input().split())
main_dishes = list(map(int, input().split()))
side_dishes = list(map(int, input().split()))

# Sort main and side dishes by price in descending order
main_dishes_sorted = [(main_dishes[i-1], i) for i in range(1, N+1)]
main_dishes_sorted.sort(reverse=True)
side_dishes_sorted = [(side_dishes[i-1], i) for i in range(1, M+1)]
side_dishes_sorted.sort(reverse=True)

# Store the incompatible pairs
incompatible_for_main = [set() for _ in range(N+1)]
for _ in range(L):
    c, d = map(int, input().split())
    incompatible_for_main[c].add(d)

# For each main dish, find the most expensive compatible side dish
max_price = 0
for main_price, i in main_dishes_sorted:
    incompatible_sides = incompatible_for_main[i]
    for side_price, j in side_dishes_sorted:
        if j not in incompatible_sides:
            max_price = max(max_price, main_price + side_price)
            break

print(max_price)