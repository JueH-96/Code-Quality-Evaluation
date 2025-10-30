n, x = map(int, input().split())
a = list(map(int, input().split()))
min_k = -1

for k in range(0, 101):
    combined = a + [k]
    combined_sorted = sorted(combined)
    total = sum(combined_sorted[1:-1])
    if total >= x:
        if min_k == -1 or k < min_k:
            min_k = k

print(min_k if min_k != -1 else -1)