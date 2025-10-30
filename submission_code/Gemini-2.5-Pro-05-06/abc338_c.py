import math

def solve():
    N = int(input())
    Q_list = list(map(int, input().split()))
    A_list = list(map(int, input().split()))
    B_list = list(map(int, input().split()))

    max_total_servings = 0

    # Determine the maximum number of servings of dish A that can be made
    # if we *only* make dish A. This gives an upper bound for iterating num_A.
    limit_num_A = math.inf
    # The problem guarantees "There is an i such that A_i >= 1".
    # This means at least one A_list[i] is positive, so limit_num_A will be updated
    # to a finite integer value.
    for i in range(N):
        if A_list[i] > 0:
            # Integer division gives how many times A_list[i] fits into Q_list[i]
            limit_num_A = min(limit_num_A, Q_list[i] // A_list[i])
    
    # If limit_num_A remained math.inf, it would mean all A_list[i] are 0.
    # This is ruled out by problem constraints.
    # limit_num_A is now a non-negative integer (it could be 0).
    # The loop for num_A iterates from 0 up to this limit (inclusive).
    # range(limit_num_A + 1) handles the case limit_num_A = 0 correctly (loop runs for num_A=0).
    
    for num_A in range(limit_num_A + 1):
        # Calculate remaining ingredients after making num_A servings of dish A.
        # Q_remaining_list stores the quantity of each ingredient left.
        Q_remaining_list = [0] * N
        
        # This check for possibility is technically redundant because num_A is bounded by limit_num_A.
        # If num_A <= limit_num_A, then for any A_list[k] > 0,
        # num_A * A_list[k] <= (Q_list[k] // A_list[k]) * A_list[k] <= Q_list[k].
        # If A_list[k] == 0, then num_A * A_list[k] = 0 <= Q_list[k].
        # So, Q_list[i] - num_A * A_list[i] will always be non-negative.
        for i in range(N):
            amount_A_needed = num_A * A_list[i]
            Q_remaining_list[i] = Q_list[i] - amount_A_needed
        
        # Calculate maximum number of dish B servings possible with Q_remaining_list.
        num_B = math.inf
        # The problem guarantees "There is an i such that B_i >= 1".
        # This means at least one B_list[i] is positive, so num_B will be updated
        # to a finite integer value.
        for i in range(N):
            if B_list[i] > 0:
                num_B = min(num_B, Q_remaining_list[i] // B_list[i])
        
        # If num_B remained math.inf (e.g. all B_list[i] were 0), this is ruled out.
        # num_B is now a non-negative integer.
        current_total_servings = num_A + num_B
        
        if current_total_servings > max_total_servings:
            max_total_servings = current_total_servings

    print(max_total_servings)

solve()