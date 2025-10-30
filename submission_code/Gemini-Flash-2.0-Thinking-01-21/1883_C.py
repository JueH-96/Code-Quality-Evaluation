def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    product = 1
    for x in a:
        product *= x
        
    if product % k == 0:
        print(0)
        return
        
    if k == 2:
        print(1)
    elif k == 3:
        has_rem_2 = False
        for x in a:
            if x % 3 == 2:
                has_rem_2 = True
                break
        if has_rem_2:
            print(1)
        else:
            print(2)
    elif k == 4:
        v2_sum = 0
        for x in a:
            v2_count = 0
            while x > 0 and x % 2 == 0:
                v2_count += 1
                x //= 2
            v2_sum += v2_count
            
        if v2_sum >= 2:
            print(0)
        elif v2_sum == 1:
            if n == 1:
                print(2)
            else:
                print(1)
        else: # v2_sum == 0
            has_rem_3_mod_4 = False
            for x in a:
                if x % 4 == 3:
                    has_rem_3_mod_4 = True
                    break
            if has_rem_3_mod_4:
                print(1)
            else:
                print(2)
    elif k == 5:
        min_ops = float('inf')
        for i in range(n):
            if a[i] % 5 != 0:
                ops = (5 - (a[i] % 5)) % 5
                min_ops = min(min_ops, ops)
        print(min_ops)

t = int(input())
for _ in range(t):
    solve()