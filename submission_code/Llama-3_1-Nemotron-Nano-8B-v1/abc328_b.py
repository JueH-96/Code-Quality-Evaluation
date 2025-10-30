def is_repdigit(n):
    s = str(n)
    return all(c == s[0] for c in s)

n = int(input())
d = list(map(int, input().split()))

count = 0

for i in range(1, n + 1):
    if is_repdigit(i):
        max_day = d[i - 1]
        for j in range(1, max_day + 1):
            if is_repdigit(j):
                count += 1

print(count)