n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

for j in range(m):
    deliciousness = b[j]
    eaten_by = -1
    
    for i in range(n):
        if deliciousness >= a[i]:
            eaten_by = i + 1  # 1-indexed
            break
    
    print(eaten_by)