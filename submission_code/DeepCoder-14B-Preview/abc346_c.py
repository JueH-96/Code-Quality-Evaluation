n, k = map(int, input().split())
a = list(map(int, input().split()))

present = set()
for num in a:
    if 1 <= num <= k:
        present.add(num)

sum_present = sum(present)
total = k * (k + 1) // 2
print(total - sum_present)