# YOUR CODE HERE
import sys

def solve():
    # Read the number of pairs, N
    N = int(sys.stdin.readline())
    
    # Initialize lists to store L_i and R_i values
    L_values = []
    R_values = []
    
    # Read N pairs of (L_i, R_i)
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        L_values.append(l)
        R_values.append(r)
    
    # Initialize the sequence X by setting X_i = L_i for all i.
    # This represents the sequence with the minimum possible sum.
    current_X = list(L_values) 
    
    # Calculate the sum of all L_i (minimum possible sum)
    sum_L = sum(L_values)
    # Calculate the sum of all R_i (maximum possible sum)
    sum_R = sum(R_values)
    
    # Check the feasibility condition:
    # If the minimum possible sum (sum_L) is already greater than 0,
    # or the maximum possible sum (sum_R) is already less than 0,
    # then it's impossible to achieve a sum of 0.
    if sum_L > 0 or sum_R < 0:
        print("No")
        return
        
    # If we reach here, it means sum_L <= 0 <= sum_R.
    # This guarantees that a solution exists.
    # We need to increase the current sum (which is sum_L) to 0.
    # The 'needed_increase' is the amount by which we must raise the total sum.
    # Since sum_L <= 0, needed_increase will be non-negative.
    needed_increase = -sum_L 
    
    # Iterate through the elements of X and adjust them greedily
    # to distribute the 'needed_increase'.
    for i in range(N):
        # If we have already increased the total sum to 0, no more adjustments are needed.
        if needed_increase == 0:
            break 
        
        # Calculate how much the current element X[i] can be increased
        # from its initial L_i value up to its R_i bound.
        possible_increase_for_i = R_values[i] - L_values[i]
        
        # The actual increase applied to X[i] is the minimum of:
        # 1. The remaining 'needed_increase' for the entire sum.
        # 2. The maximum 'possible_increase_for_i' for the current element.
        actual_increase_for_i = min(needed_increase, possible_increase_for_i)
        
        # Apply the calculated increase to X[i]
        current_X[i] += actual_increase_for_i
        
        # Reduce the total 'needed_increase' by the amount applied to X[i].
        needed_increase -= actual_increase_for_i
        
    # After the loop, the total sum of current_X will be 0.
    # Print "Yes" as a solution has been found.
    print("Yes")
    # Print the elements of the constructed sequence, space-separated.
    # The '*' operator unpacks the list into separate arguments for print().
    print(*(current_X))

# Call the solve function to execute the program.
solve()