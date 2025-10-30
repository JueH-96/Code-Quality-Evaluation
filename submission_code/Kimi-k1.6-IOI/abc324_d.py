n = int(input())
s = input().strip()

# Precompute the target digit counts
target = [0] * 10
for c in s:
    target[int(c)] += 1
target = tuple(target)

max_num = 10 ** n - 1
max_i = int(max_num ** 0.5)

count = 0

for i in range(max_i + 1):
    square = i * i
    s_square = str(square).zfill(n)
    cnt = [0] * 10
    for c in s_square:
        cnt[int(c)] += 1
    if tuple(cnt) == target:
        count += 1

print(count)