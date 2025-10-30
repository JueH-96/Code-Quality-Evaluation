def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    if k == 2:
        ans = 0
        prod = 1
        for x in a:
            prod *= x
        if prod % 2 == 0:
            print(0)
            return
        else:
            print(1)
            return
    elif k == 3:
        ans = 0
        prod = 1
        for x in a:
            prod *= x
        if prod % 3 == 0:
            print(0)
            return
        else:
            print(1)
            return
    elif k == 4:
        ans = float('inf')
        for i in range(n):
            temp_a = a[:]
            temp_a[i] += 1
            prod = 1
            for x in temp_a:
                prod *= x
            if prod % 4 == 0:
                ans = min(ans, 1)
        
        for i in range(n):
            for j in range(i + 1, n):
                temp_a = a[:]
                temp_a[i] += 1
                temp_a[j] += 1
                prod = 1
                for x in temp_a:
                    prod *= x
                if prod % 4 == 0:
                    ans = min(ans, 2)
        
        prod = 1
        for x in a:
            prod *= x
        if prod % 4 == 0:
            ans = min(ans, 0)
        
        print(ans)
        return
    elif k == 5:
        ans = 0
        prod = 1
        for x in a:
            prod *= x
        if prod % 5 == 0:
            print(0)
            return
        else:
            print(1)
            return

t = int(input())
for _ in range(t):
    solve()