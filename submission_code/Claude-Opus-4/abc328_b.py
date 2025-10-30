# YOUR CODE HERE
def is_repdigit(n):
    s = str(n)
    return all(digit == s[0] for digit in s)

N = int(input())
D = list(map(int, input().split()))

count = 0
for i in range(1, N + 1):
    if is_repdigit(i):
        for j in range(1, D[i-1] + 1):
            if is_repdigit(j):
                count += 1

print(count)