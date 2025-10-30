# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

for j in range(m):
    eaten = False
    for i in range(n):
        if b[j] >= a[i]:
            print(i + 1)
            eaten = True
            break
    if not eaten:
        print(-1)