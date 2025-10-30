def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    if k == 2:
        ans = 0
        for x in a:
            if x % 2 == 0:
                print(0)
                return
        print(1)
        return

    if k == 3:
        ans = float('inf')
        for i in range(n):
            cnt = 0
            prod = 1
            for j in range(n):
                if i == j:
                    prod *= (a[j] + 0)
                else:
                    prod *= a[j]
            
            if prod % 3 == 0:
                ans = min(ans, 0)
            else:
                
                prod = 1
                for j in range(n):
                    if i == j:
                        prod *= (a[j] + 1)
                    else:
                        prod *= a[j]
                if prod % 3 == 0:
                    ans = min(ans, 1)
                else:
                    prod = 1
                    for j in range(n):
                        if i == j:
                            prod *= (a[j] + 2)
                        else:
                            prod *= a[j]
                    if prod % 3 == 0:
                        ans = min(ans, 2)

        
        
        cnt = 0
        for x in a:
            if x % 3 == 0:
                print(0)
                return
            elif x % 3 == 2:
                cnt += 1
        
        if cnt >= 1:
            print(1)
        else:
            print(2)
        
        return
    
    if k == 4:
        ans = float('inf')
        cnt2 = 0
        for x in a:
            if x % 4 == 0:
                print(0)
                return
            elif x % 2 == 0:
                cnt2 += 1
        
        if cnt2 >= 1:
            print(1)
            return
        
        cnt3 = 0
        for x in a:
            if x % 4 == 3:
                cnt3 += 1
        
        if cnt3 >= 2:
            print(2)
            return
        
        cnt1 = 0
        for x in a:
            if x % 2 == 1:
                cnt1 += 1
        
        print(min(2, cnt1))
        return

    if k == 5:
        ans = float('inf')
        for i in range(n):
            for inc in range(5):
                cnt = 0
                prod = 1
                for j in range(n):
                    if i == j:
                        prod *= (a[j] + inc)
                    else:
                        prod *= a[j]
                if prod % 5 == 0:
                    ans = min(ans, inc)
        
        
        cnt = 0
        for x in a:
            if x % 5 == 0:
                print(0)
                return
        
        print(min(ans, 5))
        return

t = int(input())
for _ in range(t):
    solve()