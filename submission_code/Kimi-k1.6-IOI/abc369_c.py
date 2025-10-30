n = int(input())
a = list(map(int, input().split()))

if n == 1:
    print(1)
else:
    d = []
    for i in range(n-1):
        d.append(a[i+1] - a[i])
    
    x = 0
    if d:
        current_d = d[0]
        current_len = 1
        for num in d[1:]:
            if num == current_d:
                current_len += 1
            else:
                x += current_len * (current_len - 1) // 2
                current_d = num
                current_len = 1
        x += current_len * (current_len - 1) // 2
    
    ans = n + (n - 1) + x
    print(ans)