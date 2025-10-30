def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                # For two numbers x and y, x^y = y^x if:
                # 1. x = y, or
                # 2. x = 2 and y = 4, or
                # 3. x = 4 and y = 2
                # Since we're dealing with powers of 2, we need to check if 2^a[i] and 2^a[j] satisfy this
                
                # Case 1: If powers are equal
                if a[i] == a[j]:
                    count += 1
                # Case 2 & 3: If one is double of other
                elif (a[i] == 2 and a[j] == 4) or (a[i] == 4 and a[j] == 2):
                    count += 1
                    
        print(count)

solve()