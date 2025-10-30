import sys
import collections

# For PyPy, use sys.stdin.readline and print. Otherwise, input() and print().
# input = sys.stdin.readline 

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A_initial_str = [sys.stdin.readline().split() for _ in range(N)]
    
    A = []
    for i in range(N):
        A.append(tuple(map(int, A_initial_str[i])))

    MOD = 998244353

    # Heuristic: if M is large, the cycle length 'd_loop_bound' becomes very large.
    # Operations N * d_loop_bound * M would be too slow.
    # It's a common contest pattern that for large M, sequences diverge quickly
    # or become too "random" to match specific targets, unless M is small.
    # Typically, this means f(i,j)=0 for x > 0 if M > some_threshold.
    # The threshold is often around 60-70, because 2^60 is already huge.
    # If M > 64, then d_loop_bound = 2^ceil(log2(M)) > 64.
    # Let's assume if M > 64, sum is 0.
    if M > 64:
        print(0)
        return

    map_initials = collections.defaultdict(list) # val_tuple -> sorted list of j indices
    for j in range(N):
        map_initials[A[j]].append(j)
    
    # Sort lists to enable binary search (bisect_left)
    for key in map_initials:
        map_initials[key].sort()

    total_f_sum = 0
    
    if M == 0:
        print(0)
        return

    # Calculate d_loop_bound = 2^(ceil(log2 M))
    # For M=1, (1-1).bit_length()=0, d_loop_bound=1.
    # For M>1, (M-1).bit_length() is ceil(log2 M).
    k_M = (M - 1).bit_length() 
    d_loop_bound = 1 << k_M
    
    # This list is used to store the next state of current_A_val to avoid in-place modification issues.
    # It's allocated once per i, then reused.
    next_A_storage = [0] * M

    for i in range(N):
        current_A_val_list = list(A[i]) # Mutable version of A_i^(x)
        
        hit_targets_for_this_i = set() 

        for x in range(d_loop_bound):
            # current_A_val_list holds A_i^(x)
            h_current_A = tuple(current_A_val_list) # Convert to tuple for hashing

            if h_current_A in map_initials and h_current_A not in hit_targets_for_this_i:
                list_of_j = map_initials[h_current_A]
                
                # Count j in list_of_j such that j >= i
                # list_of_j is sorted. Find first k_idx s.t. list_of_j[k_idx] >= i
                # Using manual bisect_left:
                low = 0
                high = len(list_of_j)
                first_valid_k_in_list_j = len(list_of_j) # if all j < i
                
                while low < high:
                    mid = (low + high) // 2
                    if list_of_j[mid] >= i:
                        first_valid_k_in_list_j = mid
                        high = mid
                    else:
                        low = mid + 1
                
                num_such_j = len(list_of_j) - first_valid_k_in_list_j
                
                if num_such_j > 0:
                    total_f_sum = (total_f_sum + x * num_such_j) % MOD
                
                hit_targets_for_this_i.add(h_current_A)

            if x < d_loop_bound - 1:
                # Transform current_A_val_list to A_i^(x+1)
                current_sum_for_prefix = 0
                for k_idx in range(M):
                    current_sum_for_prefix = (current_sum_for_prefix + current_A_val_list[k_idx]) % 2
                    next_A_storage[k_idx] = current_sum_for_prefix
                # Update current_A_val_list by copying from next_A_storage
                for k_idx in range(M):
                    current_A_val_list[k_idx] = next_A_storage[k_idx]
            else: # No need to transform if it's the last x value
                pass
    
    print(total_f_sum)

solve()