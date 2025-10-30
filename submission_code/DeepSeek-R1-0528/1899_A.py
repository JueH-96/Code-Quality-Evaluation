def main():
    t = int(input().strip())
    dp0 = [[False] * 10 for _ in range(3)]
    dp1 = [[False] * 10 for _ in range(3)]
    
    for k in range(9, -1, -1):
        for r in range(3):
            if k == 9:
                if (r + 1) % 3 == 0 or (r + 2) % 3 == 0:
                    dp0[r][k] = True
                else:
                    dp0[r][k] = False
            else:
                s1 = (r + 1) % 3
                s2 = (r + 2) % 3
                if s1 == 0 or s2 == 0:
                    dp0[r][k] = True
                else:
                    if dp1[s1][k + 1] or dp1[s2][k + 1]:
                        dp0[r][k] = True
                    else:
                        dp0[r][k] = False
        
        for r in range(3):
            t1 = (r + 1) % 3
            t2 = (r + 2) % 3
            if dp0[t1][k] and dp0[t2][k]:
                dp1[r][k] = True
            else:
                dp1[r][k] = False
                
    results = []
    for _ in range(t):
        n = int(input().strip())
        r = n % 3
        if dp0[r][0]:
            results.append("First")
        else:
            results.append("Second")
            
    for res in results:
        print(res)

if __name__ == '__main__':
    main()