import heapq
import sys

def main():
    N = int(sys.stdin.readline())
    A_orig = list(map(int, sys.stdin.readline().split()))

    if N <= 1: # Problem constraints: N >= 2. If N=1, score is 0.
        print(0)
        return
    
    nodes_val = list(A_orig) 
    
    prev_ptr = [-1] * N 
    next_ptr = [-1] * N
    
    active = [True] * N 

    for i in range(N):
        if i > 0:
            prev_ptr[i] = i - 1
        if i < N - 1:
            next_ptr[i] = i + 1

    pq = [] 

    for i in range(N - 1):
        diff = abs(nodes_val[i] - nodes_val[i+1])
        # Store as (neg_diff, left_original_idx, right_original_idx)
        # neg_diff for max-heap behavior with heapq
        heapq.heappush(pq, (-diff, i, i+1))

    total_score = 0
    num_ops_to_perform = N // 2
    
    ops_done = 0
    while ops_done < num_ops_to_perform and pq: # Added `and pq` for robustness
        
        neg_diff, l_idx, r_idx = heapq.heappop(pq)

        # Validity check for the pair (l_idx, r_idx)
        if not active[l_idx] or not active[r_idx] or next_ptr[l_idx] != r_idx:
            continue # Stale entry from PQ, discard and pick next

        # Process this valid pair
        total_score += -neg_diff # Add positive difference
        ops_done += 1

        active[l_idx] = False
        active[r_idx] = False

        p_l_idx = prev_ptr[l_idx] 
        n_r_idx = next_ptr[r_idx] 

        # Update linked list connections for P and N'
        if p_l_idx != -1: 
            next_ptr[p_l_idx] = n_r_idx
        
        if n_r_idx != -1: 
            prev_ptr[n_r_idx] = p_l_idx
        
        # If P and N' exist (are not -1 indices), they form a new adjacent pair.
        # Add this new pair to the priority queue.
        if p_l_idx != -1 and n_r_idx != -1:
            # active[p_l_idx] and active[n_r_idx] should be true by construction
            # because they were not l_idx or r_idx.
            new_diff = abs(nodes_val[p_l_idx] - nodes_val[n_r_idx])
            heapq.heappush(pq, (-new_diff, p_l_idx, n_r_idx))
            
    sys.stdout.write(str(total_score) + "
")

if __name__ == '__main__':
    main()