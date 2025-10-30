def is_repdigit(n):
    s = str(n)
    return all(c == s[0] for c in s)

n = int(input())
d = list(map(int, input().split()))

count = 0
for i in range(1, n + 1):
    for j in range(1, d[i - 1] + 1):
        if is_repdigit(i) and is_repdigit(j) and all(c == str(i)[0] for c in str(j)) :
            count += 1

print(count)