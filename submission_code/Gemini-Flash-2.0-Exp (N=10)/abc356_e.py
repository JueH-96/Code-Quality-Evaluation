def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            max_val = max(a[i], a[j])
            min_val = min(a[i], a[j])
            total_sum += max_val // min_val
            
    print(total_sum)

solve()