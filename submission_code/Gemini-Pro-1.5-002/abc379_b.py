def solve():
    n, k = map(int, input().split())
    s = list(input())
    
    ans = 0
    
    while True:
        found = False
        for i in range(n - k + 1):
            all_o = True
            for j in range(i, i + k):
                if s[j] == 'X':
                    all_o = False
                    break
            
            if all_o:
                ans += 1
                for j in range(i, i + k):
                    s[j] = 'X'
                found = True
                break
        
        if not found:
            break
            
    print(ans)

solve()