n = int(input())
a = list(map(int, input().split()))

for i in range(n-1):
    if i < n-2:
        print(a[i]*a[i+1], end=' ')
    else:
        print(a[i]*a[i+1])