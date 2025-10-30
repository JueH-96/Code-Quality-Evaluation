# YOUR CODE HERE
def is_repdigit(n):
    s = str(n)
    return all(ch == s[0] for ch in s)

N = int(input())
D = list(map(int, input().split()))

count = 0
for i in range(1, N + 1):
    for j in range(1, D[i - 1] + 1):
        if is_repdigit(i) and is_repdigit(j):
            count += 1

print(count)