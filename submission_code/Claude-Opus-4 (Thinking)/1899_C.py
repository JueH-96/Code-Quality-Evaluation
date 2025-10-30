t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    
    max_sum = a[0]
    current_sum = a[0]
    
    for i in range(1, n):
        if (a[i-1] % 2) != (a[i] % 2):  # different parity, can extend
            current_sum = max(a[i], current_sum + a[i])
        else:  # same parity, must start new
            current_sum = a[i]
        
        max_sum = max(max_sum, current_sum)
    
    print(max_sum)