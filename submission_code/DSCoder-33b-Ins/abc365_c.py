# YOUR CODE HERE

n, m = map(int, input().split())
a = list(map(int, input().split()))

a.sort()

if sum(a) <= m:
    print("infinite")
else:
    total = 0
    for i in range(n):
        total += a[i]
        if total > m:
            print(a[i-1])
            break