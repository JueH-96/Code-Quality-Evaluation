def solve():
    n = int(input())
    a = list(map(int, input().split()))

    if n == 1:
        print(0)
        return

    total_sum = sum(a)
    
    ans1 = 0
    if total_sum % n == 0:
        target = total_sum // n
        for x in a:
            if x > target:
                ans1 += x - target
        print(ans1)
    else:
        target1 = total_sum // n
        target2 = target1 + 1
        
        count2 = total_sum % n
        count1 = n - count2
        
        ans = 0
        for x in a:
            if x > target2:
                ans += x - target2
            elif x < target1:
                ans += target1 - x
        
        ans1 = 0
        for x in a:
            if x > target2:
                ans1 += x - target2
        
        ans2 = 0
        for x in a:
            if x < target1:
                ans2 += target1 - x
        
        print(max(ans1, ans2))

solve()