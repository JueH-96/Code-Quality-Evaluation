def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    Q = list(map(int, input_data[1:1+N]))
    A = list(map(int, input_data[1+N:1+2*N]))
    B = list(map(int, input_data[1+2*N:1+3*N]))
    
    # If we cannot make any serving of A alone, or B alone, the maximum might still be zero
    # But let's systematically find the max combination x+y:
    #
    # We want x*A_i + y*B_i <= Q_i for all i, with x, y >= 0 (integers).
    #
    # Approach:
    # 1) Let max_x = min( floor(Q_i / A_i) ) over all i where A_i>0. 
    #    (Because if for some i, A_i=0, it doesn't directly limit x from that ingredient.)
    # 2) For each x in [0..max_x], compute the maximum y:
    #    leftover_i = Q_i - x*A_i (this must be >=0 for all i if A_i>0, else skip)
    #    Then y <= floor( leftover_i / B_i ) for all i with B_i>0
    #    y_max = min( leftover_i // B_i for those i ), provided leftover_i >=0 for each i
    # 3) Keep track of the max of (x + y_max).
    
    # Edge cases:
    # - Possibly A_i=0 or B_i=0 for some i. This means we do not use that ingredient in that dish.
    # - We only skip x if leftover_i < 0 for any i with A_i>0.
    # - For y we only apply constraints from i with B_i>0.
    
    # Find max_x
    max_x = 10**9  # something large
    for i in range(N):
        if A[i] > 0:
            max_x = min(max_x, Q[i] // A[i])
    
    ans = 0
    
    # We gather indices of those i where B_i>0 for quick iteration
    b_pos_indices = [i for i in range(N) if B[i] > 0]
    
    for x in range(max_x + 1):
        # compute leftover for each i
        valid_for_A = True
        for i in range(N):
            # If A[i] > 0 but Q[i] - A[i]*x < 0, skip
            if A[i] > 0:
                if Q[i] - A[i]*x < 0:
                    valid_for_A = False
                    break
        if not valid_for_A:
            continue
        
        y_max = 10**15
        # For those i with B_i > 0, compute leftover and derive a bound for y
        for i in b_pos_indices:
            leftover = Q[i] - A[i]*x
            # leftover >= 0 because we checked valid_for_A
            possible_y = leftover // B[i]
            if possible_y < y_max:
                y_max = possible_y
        
        # x + y_max might be the best so far
        if x + y_max > ans:
            ans = x + y_max
    
    print(ans)

if __name__ == "__main__":
    main()