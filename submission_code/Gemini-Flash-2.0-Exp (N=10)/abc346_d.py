def solve():
    n = int(input())
    s = input()
    c = list(map(int, input().split()))
    
    ans = float('inf')
    
    for i in range(1 << n):
        temp_s = list(s)
        cost = 0
        for j in range(n):
            if (i >> j) & 1:
                cost += c[j]
                temp_s[j] = '1' if temp_s[j] == '0' else '0'
        
        temp_s = "".join(temp_s)
        
        count = 0
        for k in range(n - 1):
            if temp_s[k] == temp_s[k+1]:
                count += 1
        
        if count == 1:
            ans = min(ans, cost)
            
    print(ans)

solve()