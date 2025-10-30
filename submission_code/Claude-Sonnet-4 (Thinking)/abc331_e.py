N, M, L = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

forbidden = set()
for _ in range(L):
    c, d = map(int, input().split())
    forbidden.add((c-1, d-1))  # Convert to 0-indexed

# Sort side dishes in descending order of cost
side_dishes_sorted = sorted(range(M), key=lambda x: b[x], reverse=True)

max_cost = 0
for main_idx in range(N):
    for side_idx in side_dishes_sorted:
        if (main_idx, side_idx) not in forbidden:
            max_cost = max(max_cost, a[main_idx] + b[side_idx])
            break

print(max_cost)