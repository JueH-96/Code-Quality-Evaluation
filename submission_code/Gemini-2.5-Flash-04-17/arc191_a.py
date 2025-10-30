import sys

def solve():
    # Read N and M
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])

    # Read strings S and T
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Use a list for mutable string S
    s_list = list(S) 

    # Perform M operations
    for k in range(M):
        char_t = T[k]

        # Find the minimum value among digits in s_list that are less than char_t
        min_val_to_improve = '9' + '1' # Initialize with a value larger than any digit '1'-'9'
        
        for i in range(N):
            if s_list[i] < char_t:
                min_val_to_improve = min(min_val_to_improve, s_list[i])

        index_to_replace = -1

        # If we found any digit in s_list < char_t (i.e., min_val_to_improve was updated)
        if min_val_to_improve <= '9':
            # Find the rightmost index containing this minimum improvable value
            rightmost_min_improvable_index = -1
            
            # Iterate from right to left to find the rightmost index efficiently
            for i in range(N - 1, -1, -1):
                 # Check if the digit matches the minimum improvable value
                 # and is indeed less than char_t (redundant check due to min_val_to_improve definition, but harmless)
                 if s_list[i] == min_val_to_improve and s_list[i] < char_t:
                     rightmost_min_improvable_index = i
                     break # Found the rightmost
                     
            index_to_replace = rightmost_min_improvable_index

        else:
            # No character in s_list is strictly less than char_t.
            # The sample 1 trace logic implies that when an improvement is possible,
            # we pick the rightmost among the minimum improvable digits.
            # The 'else' case here handles when no improvement is possible (all s_list[i] >= char_t).
            # In this scenario, the sample trace doesn't provide a direct example.
            # A plausible strategy when no improvement is possible is to replace the leftmost digit (index 0),
            # although replacing the rightmost might minimize the value decrease.
            # Based on common greedy strategies for maximizing numbers from left,
            # the leftmost position is the most critical. If we must replace without improvement,
            # perhaps replace the leftmost to keep later positions potentially larger.
            # However, following the structure derived from the sample where *some* index is always chosen based on condition:
            # The condition `if min_val_to_improve <= '9'` being false means no `s_list[i] < char_t`.
            # In this specific case, the interpretation from sample 1 breaks down.
            # Let's stick to the simplest interpretation if no improvement is possible: replace index 0.
            index_to_replace = 0 


        # Perform the replacement
        s_list[index_to_replace] = char_t

    # Print the result
    print("".join(s_list))

solve()