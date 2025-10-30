def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = sum(a)
    m = total_sum // n
    r = total_sum % n
    
    b = []
    for i in range(n - r):
        b.append(m)
    for i in range(r):
        b.append(m + 1)
        
    suffix_sum_a = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum_a[i] = suffix_sum_a[i+1] + a[i]
        
    suffix_sum_b = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum_b[i] = suffix_sum_b[i+1] + b[i]
        
    possible = True
    for k in range(1, n):
        if suffix_sum_a[k] < suffix_sum_b[k]:
            possible = False
            break
            
    if possible:
        print("Yes")
    else:
        print("No")

t = int(input())
for _ in range(t):
    solve()