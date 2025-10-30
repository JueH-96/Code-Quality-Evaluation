def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N+1]))
    
    # We need to determine if we can make all A_i + B_i equal to some constant C
    
    # First, let's calculate the pairs where neither A_i nor B_i are -1
    fixed_sums = []
    flexible_a = []
    flexible_b = []
    
    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            fixed_sums.append(A[i] + B[i])
        elif A[i] == -1 and B[i] != -1:
            flexible_a.append((B[i], 'B'))
        elif A[i] != -1 and B[i] == -1:
            flexible_b.append((A[i], 'A'))
        else:
            # Both are -1
            flexible_a.append((-1, 'A'))
            flexible_b.append((-1, 'B'))
    
    if fixed_sums:
        target_sum = min(fixed_sums)
    else:
        target_sum = 0  # We can choose any target sum if all are flexible
    
    # Check if all fixed sums can match the target sum
    for sum_val in fixed_sums:
        if sum_val != target_sum:
            print("No")
            return
    
    # Now we need to check if we can assign values to flexible_a and flexible_b to match target_sum
    for value, origin in flexible_a:
        if origin == 'B':
            # A[i] is flexible, B[i] = value
            # We need A[i] + value = target_sum
            # A[i] = target_sum - value
            if target_sum - value < 0:
                print("No")
                return
    
    for value, origin in flexible_b:
        if origin == 'A':
            # B[i] is flexible, A[i] = value
            # We need value + B[i] = target_sum
            # B[i] = target_sum - value
            if target_sum - value < 0:
                print("No")
                return
    
    # If all checks pass, we can assign values to make all sums equal to target_sum
    print("Yes")