# YOUR CODE HERE
def is_repdigit(n):
    s = str(n)
    return len(set(s)) == 1

n = int(input())
days = list(map(int, input().split()))

count = 0
for i in range(1, n + 1):
    if is_repdigit(i):
        for j in range(1, days[i-1] + 1):
            if is_repdigit(j):
                count += 1

print(count)