N = int(input())
D = list(map(int, input().split()))

def is_repdigit(i, j):
    s = str(i) + str(j)
    return all(c == s[0] for c in s)

count = 0
for i in range(1, N + 1):
    for j in range(1, D[i - 1] + 1):
        if is_repdigit(i, j):
            count += 1

print(count)