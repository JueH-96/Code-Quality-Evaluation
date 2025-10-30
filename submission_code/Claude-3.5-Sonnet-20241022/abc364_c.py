def solve():
    N, X, Y = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Try each length from 1 to N
    for length in range(1, N+1):
        # Try all possible combinations of dishes of this length
        dishes = list(range(N))
        from itertools import permutations
        for perm in permutations(dishes, length):
            sweet_sum = 0
            salt_sum = 0
            valid = True
            
            # Check each prefix of the permutation
            for i in range(length):
                sweet_sum += A[perm[i]]
                salt_sum += B[perm[i]]
                
                # If we exceed either limit before the last dish
                if i < length-1 and (sweet_sum > X or salt_sum > Y):
                    valid = False
                    break
            
            # Check if this arrangement works
            if valid and (sweet_sum > X or salt_sum > Y):
                print(length)
                return
    
    # If we get here, we need all N dishes
    print(N)

solve()