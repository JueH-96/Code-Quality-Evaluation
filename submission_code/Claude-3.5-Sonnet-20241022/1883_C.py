def solve():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        a = list(map(int, input().split()))
        
        # Calculate current product
        prod = 1
        for x in a:
            prod *= x
            
        if k == 2:
            print(0 if prod % 2 == 0 else 1)
            continue
            
        if k == 3:
            if prod % 3 == 0:
                print(0)
            elif prod % 3 == 2:
                print(1)
            else:
                print(2)
            continue
            
        if k == 4:
            # Count factors of 2
            twos = 0
            p = prod
            while p % 2 == 0:
                twos += 1
                p //= 2
                
            if twos >= 2:
                print(0)
            elif twos == 1:
                print(1)
            else:
                print(2)
            continue
            
        if k == 5:
            rem = prod % 5
            if rem == 0:
                print(0)
            else:
                # Find minimum steps to next multiple of 5
                steps = 5 - rem
                print(steps)

solve()