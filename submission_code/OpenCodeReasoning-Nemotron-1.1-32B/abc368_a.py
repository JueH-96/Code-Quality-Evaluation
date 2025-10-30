n, k = map(int, input().split())
arr = list(map(int, input().split()))
result = arr[n-k:] + arr[:n-k]
print(" ".join(map(str, result)))