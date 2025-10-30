import sys

# Increase recursion depth for potentially large N, though not strictly needed for this problem
# sys.setrecursionlimit(200005)

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read pairs (L_i, R_i) and calculate sum of L_i and sum of R_i
    LR = []
    S_L = 0
    S_R = 0
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        LR.append((l, r))
        S_L += l
        S_R += r

    # Check if a solution exists
    # A solution exists if and only if sum of minimums <= 0 <= sum of maximums
    # We are given L_i <= R_i, so S_L <= S_R is always true.
    # The necessary and sufficient condition for the existence of X is S_L <= 0 <= S_R.
    if S_L > 0 or S_R < 0:
        print("No")
    else:
        print("Yes")
        
        # Construct a solution
        # Start with the minimum possible values for each X_i, which is L_i
        X = [l for l, r in LR]
        
        # The current sum is S_L. We need the sum to be 0.
        # The amount we need to increase the sum by is 0 - S_L = -S_L.
        needed_increase = -S_L

        # Greedily increase X_i values to reach the target sum 0
        # Iterate through the elements and add as much as possible up to the required increase
        # or the maximum allowed increase for that element (R_i - L_i)
        for i in range(N):
            l, r = LR[i]
            
            # The maximum we can increase X[i] from its current value L[i] is R[i] - L[i]
            max_increase_for_i = r - l
            
            # The actual increase we apply is limited by the remaining needed increase
            # and the maximum possible increase for this element.
            increase_i = min(needed_increase, max_increase_for_i)
            
            # Add the increase to X[i]
            X[i] += increase_i
            
            # Decrease the total needed increase
            needed_increase -= increase_i
            
            # If needed_increase becomes 0, we have reached the target sum.
            # Any further increases for subsequent elements will be min(0, ...) = 0,
            # so their values will remain at L_j and the total sum won't change.
            if needed_increase == 0:
                break # Optimization: stop early if sum target is met

        # After the loop, the sum of X is guaranteed to be 0 because the total
        # needed increase (-S_L) is less than or equal to the total possible increase (S_R - S_L).
        # The needed_increase variable will be 0 at the end.

        # Print the resulting sequence X
        # Use map(str, X) and join for efficient string formatting of the output line
        print(*X)

solve()