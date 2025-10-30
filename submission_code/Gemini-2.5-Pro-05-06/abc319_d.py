import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    L_arr = list(map(int, sys.stdin.readline().split()))

    # can_fit checks if words can be fit into M_val lines with window width W
    def can_fit(W, N_val, M_val, current_L_arr):
        num_lines = 1
        current_line_length = 0
        for i in range(N_val):
            word_len = current_L_arr[i]
            
            # A single word must fit within W.
            # This check is technically redundant if the binary search's 'low'
            # is initialized to max(L_arr), because then W >= max(L_arr) >= word_len.
            # However, it's a good check for the general logic of can_fit.
            if word_len > W:
                return False

            if current_line_length == 0: # First word on this line
                current_line_length = word_len
            else: # Not the first word on this line
                if current_line_length + 1 + word_len <= W:
                    # Word fits on current line (with a preceding space)
                    current_line_length += 1 + word_len
                else: 
                    # Word doesn't fit, start a new line
                    num_lines += 1
                    current_line_length = word_len # This word starts the new line
                    if num_lines > M_val: 
                        # Exceeded allowed lines, no need to continue
                        return False
        
        # If all words are placed and num_lines has not exceeded M_val
        return True

    # Binary search for minimum W
    
    # Smallest possible W is the width of the longest word.
    # L_arr is guaranteed non-empty (N >= 1).
    # L_i >= 1, so max(L_arr) >= 1.
    # If L_arr can be empty, max might need a default value or check.
    # Here, N >= 1 ensures L_arr is not empty.
    low = max(L_arr) 
    
    # Largest possible W is if all words are on one line (M >= 1 is guaranteed).
    # This is sum of all word lengths + (N-1) spaces.
    # If N=1, N-1=0, so it's sum(L_arr) = L_arr[0]. This formula works.
    high = sum(L_arr) + (N - 1) 

    ans = high # Initialize ans with a known possible W (the largest one)

    while low <= high:
        mid = low + (high - low) // 2
        # mid will always be >= low >= max(L_arr) >= 1.
        if can_fit(mid, N, M, L_arr):
            # This width 'mid' works, try for an even smaller W
            ans = mid
            high = mid - 1
        else:
            # This width 'mid' is too small, need a larger W
            low = mid + 1
            
    print(ans)

# Call the solver function
solve()