n = int(input())
h = list(map(int, input().split()))

a = [0] * (n + 1)
operations = 0
result = []
found = [False] * n

while len(result) < n:
    operations += 1
    a[0] += 1
    
    for i in range(1, n + 1):
        if a[i-1] > a[i] and a[i-1] > h[i-1]:
            a[i-1] -= 1
            a[i] += 1
    
    for i in range(n):
        if not found[i] and a[i+1] > 0:
            result.append(operations)
            found[i] = True

print(*result)