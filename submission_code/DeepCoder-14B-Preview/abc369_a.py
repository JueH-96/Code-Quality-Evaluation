n = int(input())
a = list(map(int, input().split()))

if n == 1:
    print(1)
    exit()

diff = [a[i+1] - a[i] for i in range(n-1)]
m = len(diff)

run = [0] * m
run[-1] = m - 1

for i in range(m-2, -1, -1):
    if diff[i] == diff[i+1]:
        run[i] = run[i+1]
    else:
        run[i] = i

total = 0
for l in range(n):
    if l == n - 1:
        total += 1
    else:
        total += (run[l] - l + 2)

print(total)