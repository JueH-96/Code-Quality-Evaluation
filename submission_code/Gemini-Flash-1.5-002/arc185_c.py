# YOUR CODE HERE
n, x = map(int, input().split())
a = list(map(int, input().split()))

found = False
for i in range(n - 2):
    for j in range(i + 1, n - 1):
        for k in range(j + 1, n):
            if a[i] + a[j] + a[k] == x:
                print(i + 1, j + 1, k + 1)
                found = True
                break
        if found:
            break
    if found:
        break

if not found:
    print(-1)