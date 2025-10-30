# YOUR CODE HERE
n, m = map(int, input().split())
a = sorted(map(int, input().split()))
b = sorted(map(int, input().split()))
i, j = 0, m-1
while i < n and j >= 0:
    if a[i] <= b[j]:
        break
    j -= 1
print(a[i] if i < n else b[j])