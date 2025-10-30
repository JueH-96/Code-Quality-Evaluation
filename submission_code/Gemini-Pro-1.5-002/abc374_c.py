def solve():
    n = int(input())
    k = list(map(int, input().split()))
    
    total_sum = sum(k)
    min_max_sum = float('inf')
    
    for i in range(1 << n):
        group_a_sum = 0
        for j in range(n):
            if (i >> j) & 1:
                group_a_sum += k[j]
        
        group_b_sum = total_sum - group_a_sum
        min_max_sum = min(min_max_sum, max(group_a_sum, group_b_sum))
        
    print(min_max_sum)

solve()