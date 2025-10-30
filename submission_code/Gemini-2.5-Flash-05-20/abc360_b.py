import sys

def solve():
    # Read the input strings S and T from standard input.
    # sys.stdin.readline().split() reads a line and splits it by whitespace,
    # correctly handling the newline character.
    S, T = sys.stdin.readline().split()
    
    n_s = len(S)
    n_t = len(T)

    # Iterate through all possible values for 'w'.
    # According to the problem constraints, 1 <= w < |S|.
    # In Python, range(start, end) generates numbers from 'start' up to 'end - 1'.
    # So, range(1, n_s) correctly covers w from 1 to n_s - 1.
    for w in range(1, n_s):
        # Iterate through all possible values for 'c'.
        # According to the problem constraints, 1 <= c <= w.
        # So, range(1, w + 1) correctly covers c from 1 to w.
        for c in range(1, w + 1):
            # This list will temporarily store the characters extracted from S
            # for the current (c, w) pair.
            current_T_candidate_chars = []
            
            # 'start_idx' keeps track of the beginning of the current w-length segment in S.
            start_idx = 0
            
            # Loop through S, processing it in blocks of size 'w'.
            while start_idx < n_s:
                # Extract the current segment from S.
                # Python's slicing (S[start_idx : end_idx]) handles cases where
                # 'end_idx' goes beyond the string's length gracefully,
                # taking characters up to the end of the string.
                sub = S[start_idx : start_idx + w]
                
                # Check if the current segment 'sub' is long enough to have a c-th character.
                # 'c' is 1-indexed in the problem description, so for 0-indexed Python strings,
                # we need 'c-1'. This means the segment length must be at least 'c'.
                if len(sub) >= c:
                    # Append the (c-1)-th character (0-indexed) to our list.
                    current_T_candidate_chars.append(sub[c - 1])
                
                # Optimization: If the number of characters collected so far exceeds
                # the length of T, this candidate can no longer possibly match T.
                # We can stop processing this (c, w) pair and move to the next one.
                if len(current_T_candidate_chars) > n_t:
                    break # Exit the inner 'while' loop
                
                # Move 'start_idx' to the beginning of the next 'w'-length segment.
                start_idx += w
            
            # After processing all segments for the current (c, w) pair:
            # First, check if the length of the collected characters matches the length of T.
            # If not, they cannot be equal. This also implicitly handles cases where the
            # 'while' loop was broken early due to the optimization.
            if len(current_T_candidate_chars) == n_t:
                # If lengths match, join the characters to form a string.
                result_str = "".join(current_T_candidate_chars)
                
                # Compare the formed string with the target string T.
                if result_str == T:
                    # If they are equal, we found a valid (c, w) pair.
                    # Print "Yes" and terminate the program immediately.
                    print("Yes")
                    return 

    # If the program reaches this point, it means no (c, w) pair
    # satisfying the conditions was found after checking all possibilities.
    # Print "No".
    print("No")

# Call the solve function to execute the program.
solve()