t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    arr = list(map(int, input().split()))
    best = arr[0]
    current = arr[0]
    for i in range(1, n):
        if (arr[i] % 2) != (arr[i-1] % 2):
            current = max(arr[i], current + arr[i])
        else:
            current = arr[i]
        if current > best:
            best = current
    print(best)