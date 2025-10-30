import sys

def main():
    N = int(sys.stdin.readline())
    A_str = sys.stdin.readline().split()
    # Convert string list to int list. Using map is slightly faster for large N.
    A = list(map(int, A_str))

    # last_pos[value] stores the 0-indexed position where 'value' was last seen.
    # Values in A are A_i in [1, N] as per constraints.
    # So, last_pos array needs to be of size N+1 to store indices for values 1 through N.
    # Index 0 of last_pos will be unused as values are >= 1.
    # Initialize with -1. For an element A[k], if its value has not been seen before, 
    # its prev_occurrence_idx (p_k) will be -1.
    last_pos = [-1] * (N + 1) 
    
    total_sum_f = 0
    
    # Iterate k from 0 to N-1 (0-indexed loop for the array A)
    for k in range(N):
        val = A[k] # Current value A_k
        
        # p_k is the 0-indexed position of the previous occurrence of val.
        # If val has not been seen before, last_pos[val] is -1 from initialization.
        p_k = last_pos[val]
        
        # A[k] is the first occurrence of 'val' in a subarray A[l...r] if:
        # 1. A[k] is in the subarray: l <= k <= r
        # 2. 'val' does not appear in A[l...k-1]: this means l > p_k
        # So, we need to count pairs (l,r) such that p_k < l <= k AND k <= r <= N-1.
        
        # Number of choices for l:
        # l can range from p_k + 1 to k, inclusive.
        # Count of choices for l is k - (p_k + 1) + 1 = k - p_k.
        choices_for_l = k - p_k
        
        # Number of choices for r:
        # r can range from k to N-1, inclusive.
        # Count of choices for r is (N-1) - k + 1 = N - k.
        choices_for_r = N - k
        
        # For each such pair (l,r), A[k] (as the first occurrence of 'val') 
        # contributes 1 to f(l,r).
        # So, A[k] contributes (choices_for_l * choices_for_r) to the total sum.
        contribution_of_A_k = choices_for_l * choices_for_r
        total_sum_f += contribution_of_A_k
        
        # Update the last seen position of 'val' to current index k
        last_pos[val] = k
        
    sys.stdout.write(str(total_sum_f) + "
")

if __name__ == '__main__':
    main()