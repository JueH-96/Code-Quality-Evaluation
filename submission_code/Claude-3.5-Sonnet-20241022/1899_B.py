def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    ans = 0
    for k in range(1, n + 1):
        if n % k != 0:
            continue
            
        trucks = []
        curr_sum = 0
        count = 0
        
        for i in range(n):
            curr_sum += a[i]
            count += 1
            
            if count == k:
                trucks.append(curr_sum)
                curr_sum = 0
                count = 0
        
        if trucks:
            ans = max(ans, max(trucks) - min(trucks))
    
    print(ans)

t = int(input())
for _ in range(t):
    solve()