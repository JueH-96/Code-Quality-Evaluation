def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    if k == 2:
        # Need at least 1 factor of 2
        for x in a:
            if x % 2 == 0:
                return 0
        # Need to make one number even
        return 1
    
    elif k == 3:
        # Need at least 1 factor of 3
        for x in a:
            if x % 3 == 0:
                return 0
        # Find minimum operations to make a number divisible by 3
        min_ops = float('inf')
        for x in a:
            ops = (3 - x % 3) % 3
            min_ops = min(min_ops, ops)
        return min_ops
    
    elif k == 4:
        # Need at least 2 factors of 2
        count_2 = 0
        costs = []  # cost to add one factor of 2 to each element
        
        for x in a:
            temp_count = 0
            temp_x = x
            while temp_x % 2 == 0:
                temp_count += 1
                temp_x //= 2
            count_2 += temp_count
            
            # Cost to make this number have one more factor of 2
            if x % 2 == 0:
                costs.append(2)  # multiply by 2 (add 2 to make it divisible by 4)
            else:
                costs.append(1)  # multiply by 2 (add 1 to make it even)
        
        if count_2 >= 2:
            return 0
        elif count_2 == 1:
            return min(costs)
        else:  # count_2 == 0
            costs.sort()
            return costs[0] + costs[1]
    
    elif k == 5:
        # Need at least 1 factor of 5
        for x in a:
            if x % 5 == 0:
                return 0
        # Find minimum operations to make a number divisible by 5
        min_ops = float('inf')
        for x in a:
            ops = (5 - x % 5) % 5
            min_ops = min(min_ops, ops)
        return min_ops

t = int(input())
for _ in range(t):
    print(solve())