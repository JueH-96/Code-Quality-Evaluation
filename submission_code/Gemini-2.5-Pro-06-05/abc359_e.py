# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem by simulating the process step-by-step.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        
        # Read input
        N = int(readline())
        H = list(map(int, readline().split()))

    except (IOError, ValueError):
        # Handle empty input for local testing or unexpected formats
        N = 0
        H = []

    if N == 0:
        return
        
    # ans will store the results for i=1 to N
    ans = [0] * N
    
    # t is the current number of operations (time)
    t = 0
    # A stores the values A_0, A_1, ... up to the current required length
    A = [0] 

    # We find the answer for each i from 1 to N sequentially
    for i in range(1, N + 1):
        # For each new i, we are looking for the time A_i becomes positive.
        # The state (t, A) is carried over from the previous iteration.
        
        # A_i starts at 0. We extend our array A to include it.
        if i > 1:
            A.append(0)

        # Loop through time steps until A_i > 0
        while True:
            # A_i becomes > 0 for the first time at operation t+1
            t += 1
            
            # Step 1: Increase A_0 by 1
            A[0] += 1
            
            # Step 2: Cascade of transfers for j from 1 to i
            # The cascade needs to be simulated up to index i because a transfer
            # to A_i can only happen after potential updates to A_1, ..., A_{i-1}.
            for j in range(1, i + 1):
                # We are looking for the first time A_i becomes positive.
                # This check happens at index j=i.
                if j == i:
                    # The condition for transfer from A_{i-1} to A_i.
                    # Since A_i is 0, A_{i-1} > A_i is true if A_{i-1} > 0.
                    # The condition A_{i-1} > H_{i-1} is what matters.
                    if A[j-1] > H[j-1]:
                        # Success! A_i becomes positive at time t.
                        ans[i-1] = t
                        
                        # Perform the transfer
                        A[j-1] -= 1
                        A[j] += 1
                        
                        # We found the answer for i, so break the while loop
                        # and move to the next i.
                        goto_next_i = True
                        break
                else: # j < i
                    # Regular cascade updates for indices before i
                    if A[j-1] > A[j] and A[j-1] > H[j-1]:
                        A[j-1] -= 1
                        A[j] += 1
            else:
                # This 'else' belongs to the 'for' loop.
                # It runs if the loop completes without a 'break'.
                # This means A_i is still 0.
                goto_next_i = False

            if goto_next_i:
                break
            
    print(*ans)

solve()