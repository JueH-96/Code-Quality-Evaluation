# YOUR CODE HERE
import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # If K=1, the only person gets the whole cake. Min mass is total mass. 
    # All N cut lines are never cut.
    if K == 1:
        print(sum(A), N)
        return

    total_mass = sum(A)

    # Double the array to handle circularity easily in prefix sum calculations.
    A_doubled = A + A
    
    # Compute prefix sums on the doubled array. Using 1-based indexing for prefix sums
    # makes range sum calculation slightly cleaner: P[j] - P[i-1]. Let's use 0-based sums.
    # prefix_sum_doubled[k] = sum(A_doubled[0]...A_doubled[k-1])
    prefix_sum_doubled = [0] * (2 * N + 1)
    for i in range(2 * N):
        prefix_sum_doubled[i+1] = prefix_sum_doubled[i] + A_doubled[i]

    # Check function: determines if it's possible to partition the cake into K blocks
    # such that each block has a mass of at least X.
    # This check function implements the standard greedy approach combined with binary lifting.
    # It verifies if there exists any starting piece `i` such that greedily forming minimal
    # blocks of sum >= X covers the entire cake using at most K blocks.
    def check(X):
        
        # Compute f[i] = index AFTER the end of the minimal block starting at piece i with sum >= X
        # Compute num_pieces[i] = number of pieces in this minimal block
        # These are computed based on the doubled array indices (0 to 2N-1)
        
        # Stores the index in the doubled array AFTER the end of the minimal block starting at i
        f = [0] * N 
        # Stores the number of pieces in that block
        num_pieces = [0] * N 
        
        curr_j = 0 # Use two pointers approach to find endpoints efficiently
        for i in range(N): # Only need to compute for starts 0 to N-1
             # Ensure curr_j starts at least from i
             curr_j = max(curr_j, i)

             # Target sum for the block starting at i
             # In 0-based prefix sums: Sum(A[i..j]) = prefix_sum_doubled[j+1] - prefix_sum_doubled[i]
             target_min_sum = prefix_sum_doubled[i] + X
             
             # Find smallest j >= i such that sum >= X
             # The end index j must be less than i+N because a block cannot span more than N pieces.
             while curr_j < i + N and prefix_sum_doubled[curr_j + 1] < target_min_sum:
                 curr_j += 1
             
             # If curr_j reaches i+N, it means no block starting at i with length <= N has sum >= X
             if curr_j == i + N:
                  # Check if the sum of all N pieces is >= X
                 # sum(A[i..i+N-1]) = prefix_sum_doubled[i+N] - prefix_sum_doubled[i]
                 if prefix_sum_doubled[i+N] - prefix_sum_doubled[i] < X:
                      # Cannot form even one block of size >= X starting at i within N pieces.
                      # This means X is too large.
                      return False 
                 else: 
                     # The minimal block must be exactly N pieces long.
                     # The end index is i+N-1. The index AFTER the end is i+N.
                     f[i] = i + N
                     num_pieces[i] = N
             else:
                 # Found endpoint curr_j such that block i..curr_j has sum >= X
                 # And block i..curr_j-1 has sum < X (implicit from `while` loop condition)
                 f[i] = curr_j + 1
                 num_pieces[i] = (curr_j + 1) - i
        
        # Binary lifting approach to check feasibility efficiently.
        # We need to determine if there exists a start index `i` (0 to N-1) such that
        # taking K minimal blocks greedily covers at least N pieces total.
        
        # Maximum power of 2 needed for K steps
        MAX_LOG_K = K.bit_length() 
        
        # lift_g[k][i] stores the next starting piece index (modulo N) after 2^k greedy steps from piece i.
        lift_g = [[0] * N for _ in range(MAX_LOG_K)] 
        # lift_s[k][i] stores the total number of pieces covered in those 2^k steps.
        lift_s = [[0] * N for _ in range(MAX_LOG_K)] 
        
        # Base case k=0: 1 step (2^0 = 1)
        for i in range(N):
             lift_g[0][i] = f[i] % N  # The next start index wraps around modulo N
             lift_s[0][i] = num_pieces[i]

        # Compute table for k = 1 up to MAX_LOG_K - 1
        for k in range(MAX_LOG_K - 1):
            for i in range(N):
                # State after first 2^k steps starting from i
                next_idx_mod_N = lift_g[k][i]
                pieces1 = lift_s[k][i]
                
                # State after next 2^k steps starting from next_idx_mod_N
                next_next_idx_mod_N = lift_g[k][next_idx_mod_N]
                pieces2 = lift_s[k][next_idx_mod_N]
                
                # Combine states for 2^(k+1) steps
                lift_g[k+1][i] = next_next_idx_mod_N
                
                # Sum of pieces. Cap at N because if we cover >= N pieces, feasibility is met.
                # Capping simplifies check later and prevents large numbers, although Python handles them.
                if pieces1 >= N or pieces2 >= N:
                    lift_s[k+1][i] = N
                else:
                   lift_s[k+1][i] = min(N, pieces1 + pieces2) 
                   
        # Check if any start 'i' covers at least N pieces in K steps using the lifting table
        for i in range(N):
            current_idx_mod_N = i
            total_s = 0 # Total pieces covered starting from i
            
            temp_K = K # Number of steps to simulate
            
            # Iterate through bits of K from most significant to least significant
            for k in range(MAX_LOG_K - 1, -1, -1):
                if (temp_K >> k) & 1: # If the k-th bit of K is 1
                    
                    pieces_to_add = lift_s[k][current_idx_mod_N]
                    
                    # Add pieces covered in this 2^k step jump
                    total_s += pieces_to_add
                    
                    # Update current index for next jump
                    current_idx_mod_N = lift_g[k][current_idx_mod_N]

                    # If already covered N pieces, no need to continue this path for 'i'
                    if total_s >= N:
                         total_s = N # Cap at N
                         break 
            
            # Check after simulating K steps for starting index 'i'
            if total_s >= N:
                 return True # Found a starting point i that works, minimum mass X is possible

        # If no starting point works after checking all i from 0 to N-1
        return False


    # Binary search for the maximum possible minimum mass x (ans_x)
    low = 1 # Minimum possible piece mass is 1
    # Maximum possible average mass per person. Upper bound for ans_x.
    high = total_mass // K + 1 
    ans_x = 0

    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            # If minimum mass `mid` is achievable, try higher values
            ans_x = mid
            low = mid + 1
        else:
            # If `mid` is not achievable, need to aim lower
            high = mid - 1
            
    # Compute y: number of cut lines that are never cut for the optimal minimum mass x = ans_x
    
    X = ans_x # Use the found optimal minimum mass
    
    # Based on analysis and testing samples, the set of potentially cut lines seems to be
    # exactly the set of endpoints `j` (0-based index) of all *minimal* blocks {i...j} 
    # that have sum >= X, for all possible start indices i (0 to N-1).
    
    # Store the 0-based indices `j` such that piece `j` is the last piece of some minimal block
    minimal_endpoints = set() 

    curr_j = 0 # Use two pointers again for efficiency
    for i in range(N): # Check all possible start indices i
        curr_j = max(curr_j, i)
        target_min_sum = prefix_sum_doubled[i] + X

        while curr_j < i + N and prefix_sum_doubled[curr_j + 1] < target_min_sum:
             curr_j += 1
        
        # After the loop, curr_j is the end index of the minimal block (or i+N if impossible/N pieces needed)
        
        if curr_j == i + N: 
             # If the block required exactly N pieces (sum >= X)
             if prefix_sum_doubled[i+N] - prefix_sum_doubled[i] >= X:
                 # Endpoint is (i + N - 1). The cut line is after this piece.
                 minimal_endpoints.add((i + N - 1) % N) 
        else: 
             # Found endpoint curr_j. The cut line is after piece curr_j.
             minimal_endpoints.add(curr_j % N)
            
    # The number of never cut lines is N minus the number of potentially cut lines.
    ans_y = N - len(minimal_endpoints)
    
    print(ans_x, ans_y)

solve()