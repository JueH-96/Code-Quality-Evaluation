n = int(input().strip())
arr = list(map(int, input().split()))

max1 = -10**18
max2 = -10**18
idx1 = None
idx2 = None

for i in range(n):
    num = arr[i]
    if num > max1:
        max2 = max1
        idx2 = idx1
        max1 = num
        idx1 = i + 1
    elif num > max2:
        max2 = num
        idx2 = i + 1

print(idx2)