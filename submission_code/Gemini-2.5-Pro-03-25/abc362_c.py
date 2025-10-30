# YOUR CODE HERE
import sys

def solve():
    # Read the number of pairs N from standard input
    N = int(sys.stdin.readline())
    
    # Initialize lists to store the lower bounds L_i and upper bounds R_i
    L_list = []
    R_list = []
    
    # Initialize the minimum possible sum (S_min) and maximum possible sum (S_max)
    # S_min is the sum if we choose X_i = L_i for all i
    # S_max is the sum if we choose X_i = R_i for all i
    S_min = 0
    S_max = 0
    
    # Read N pairs (L_i, R_i) from standard input
    for _ in range(N):
        # Read L and R for the current pair
        L, R = map(int, sys.stdin.readline().split())
        # Append L and R to their respective lists
        L_list.append(L)
        R_list.append(R)
        # Update the minimum and maximum possible sums
        S_min += L
        S_max += R

    # A sequence X satisfying the conditions exists if and only if
    # the target sum 0 is within the range of possible sums [S_min, S_max].
    # Check this necessary and sufficient condition.
    if not (S_min <= 0 <= S_max):
        # If 0 is outside the possible range [S_min, S_max], no solution exists.
        # Print "No" and return.
        print("No")
        return

    # If the condition S_min <= 0 <= S_max holds, a solution exists.
    # Print "Yes" to indicate a solution will be provided.
    print("Yes")
    
    # We will construct a valid sequence X using a greedy approach.
    # Start by initializing the sequence X with the lower bounds: X_i = L_i for all i.
    # The sum of this initial sequence is S_min.
    X = list(L_list) 
    
    # Calculate the total amount by which the sum needs to be increased
    # to reach the target sum of 0.
    # The current sum is S_min. The target sum is 0.
    # The required increase = Target Sum - Current Sum = 0 - S_min = -S_min.
    # Since we know S_min <= 0, the needed_increase is non-negative.
    needed_increase = -S_min 
    
    # Iterate through the elements X_i (from i=0 to N-1) and distribute the needed_increase greedily.
    for i in range(N):
        # Optimization: If the needed increase has become 0 (or less, theoretically),
        # it means we have reached the target sum of 0. We can stop distributing the increase.
        if needed_increase <= 0:
             break

        # Calculate the maximum possible increase for the current element X[i].
        # Since X[i] is currently L_i, the maximum value it can take is R_i.
        # So, the maximum possible increase is R_i - L_i.
        max_increase_i = R_list[i] - L_list[i]
        
        # Determine the actual amount to increase X[i] by.
        # We should increase X[i] by as much as possible, but not more than
        # what's needed overall (needed_increase) and not more than what's allowed
        # for this element (max_increase_i).
        # So, the increase amount is the minimum of these two values.
        increase_amount = min(max_increase_i, needed_increase)
        
        # Apply the calculated increase to the current element X[i].
        X[i] += increase_amount
        
        # Reduce the total needed increase by the amount that was just applied.
        needed_increase -= increase_amount

    # After the loop finishes, the total sum of the elements in X is guaranteed to be 0.
    # This is because the condition S_min <= 0 <= S_max ensured that the total needed increase
    # (-S_min) is less than or equal to the total maximum possible increase (S_max - S_min).
    # The greedy approach distributes exactly needed_increase amount.
    
    # Print the resulting sequence X. The elements should be space-separated.
    # The * operator unpacks the list elements, passing them as separate arguments to print,
    # which prints them space-separated by default.
    print(*(X))

# Call the solve function to execute the program logic
solve()