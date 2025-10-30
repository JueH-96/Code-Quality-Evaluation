n = int(input())
a = list(map(int, input().split()))

if n == 0:
    print(0)
else:
    prefix_min = [0] * n
    prefix_min[0] = a[0] - 0
    for i in range(1, n):
        prefix_min[i] = min(prefix_min[i-1], a[i] - i)
    
    suffix_min = [0] * n
    suffix_min[-1] = a[-1] + (n-1)
    for i in range(n-2, -1, -1):
        suffix_min[i] = min(suffix_min[i+1], a[i] + i)
    
    max_k = 0
    for j in range(n):
        left = j + prefix_min[j]
        right = suffix_min[j] - j
        d_max = min(j, (n-1) - j)
        current = min(left, right, d_max + 1)
        if current > max_k:
            max_k = current
    
    print(max_k)