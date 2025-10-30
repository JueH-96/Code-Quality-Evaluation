# YOUR CODE HERE
import sys

def solve():
    # Read input values for N, M, C, K
    N, M, C, K = map(int, sys.stdin.readline().split())
    # Read the integer sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # Handle the special case C=0
    # If C is 0, (Ck + A_i) mod M simplifies to A_i mod M.
    # Since 0 <= A_i < M, this is just A_i.
    # The minimum over i is constant for all k.
    if C == 0:
        # Check if N is 0, although constraints state N >= 1.
        if N == 0: 
             print(0)
             return
        
        # Find the minimum value in A
        min_A = A[0]
        for i in range(1, N):
            if A[i] < min_A:
                min_A = A[i]
        
        # The total sum is K times this minimum value.
        # Python's arbitrary precision integers handle potentially large results.
        print(K * min_A) 
        return

    # Create a mutable copy of the list A to store the current state
    current_A = A[:] 
    
    total_sum = 0
    k_total = 0 # Tracks the number of steps k processed so far

    # Main loop: process steps in blocks until K steps are completed
    while k_total < K:
        
        # Find the minimum and maximum values in the current list current_A
        # This takes O(N) time per iteration.
        # Constraints state N >= 1, so the list is never empty.
        
        y_min = current_A[0]
        y_max = current_A[0]
        for i in range(1, N):
            val = current_A[i]
            if val < y_min:
                y_min = val
            if val > y_max:
                y_max = val

        # Calculate the number of steps (k0) until the current maximum element (y_max)
        # would reach or exceed M if C was repeatedly added. This determines when the next
        # wrap-around event *might* occur (specifically for the element that is currently maximum).
        # If y_max is already close to M (>= M-C), it will wrap around in the next step (k0=1).
        
        # The threshold value is M-C. Elements >= M-C wrap around when C is added.
        if y_max < M - C:
             # Calculate ceil((M - y_max) / C).
             # This represents the number of steps required for y_max to reach or exceed M.
             # Use integer division formula for ceiling: ceil(X / Y) = (X + Y - 1) // Y for positive Y.
             # C > 0 is guaranteed because C=0 case is handled separately.
             k0 = (M - y_max + C - 1) // C
        else:
             # If y_max >= M - C, it wraps around in 1 step.
             k0 = 1

        # Determine the actual number of steps to process in this block.
        # This is the minimum of k0 (steps until next potential wrap) and the remaining steps needed (K - k_total).
        steps_to_take = min(k0, K - k_total)

        # If no steps can be taken (e.g., k_total already reached K), finish.
        if steps_to_take <= 0: 
             break

        # Calculate the sum contributed by this block of steps.
        # The minimum value sequence for these steps is y_min, y_min+C, y_min+2C, ...
        # This forms an arithmetic progression. The sum is calculated using the standard formula:
        # Sum = k * first_term + C * k * (k - 1) / 2
        # where k is the number of steps (steps_to_take).
        k_ = steps_to_take
        
        # Calculate the second term C * k_ * (k_ - 1) / 2.
        # Python integers handle arbitrary sizes, preventing overflow.
        term2_val = (C * k_ * (k_ - 1)) // 2
        
        # Calculate the total sum for the current block
        current_block_sum = (k_ * y_min + term2_val)
        
        # Add this block's sum to the overall total sum
        total_sum += current_block_sum

        # Update the total number of steps processed
        k_total += steps_to_take

        # If all K steps have been processed, we can exit the loop before updating the state.
        if k_total >= K:
            break

        # Update the state of current_A for the next iteration.
        # Apply the transformation (x + steps_to_take * C) mod M to all elements.
        # The value V = steps_to_take * C can be very large.
        V = steps_to_take * C
        
        # Use the property (a + b) % m = (a % m + b % m) % m to handle large V efficiently.
        # Since current_A elements x are already in [0, M-1], x % M = x.
        # So we need (x + V % M) % M.
        V_mod_M = V % M
        
        # Update each element in the list current_A (O(N) time).
        for i in range(N):
           current_A[i] = (current_A[i] + V_mod_M) % M
           
    # Print the final computed total sum
    print(total_sum)

# Execute the solve function
solve()