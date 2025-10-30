import sys

def solve():
    # Read input strings S and T from standard input.
    S, T = sys.stdin.readline().split()
    
    # Get the lengths of strings S and T.
    N = len(S)
    M = len(T)

    # Iterate through possible values of w.
    # The problem constraint is 1 <= w < |S|. In Python, this translates to
    # w ranging from 1 up to N-1 (inclusive).
    for w in range(1, N):
        # Iterate through possible values of c for the current w.
        # The problem constraint is 1 <= c <= w. In Python, this translates to
        # c ranging from 1 up to w (inclusive).
        for c in range(1, w + 1):
            # Use a list to store the characters extracted from S for the current (c, w) pair.
            # Building strings by appending to a list and then joining is generally
            # more efficient in Python than repeated string concatenation.
            result_chars = []
            
            # Iterate through the starting indices of the substrings formed by splitting S
            # at every w characters. The starting indices are 0, w, 2*w, 3*w, and so on,
            # as long as the starting index is less than the length of S.
            start = 0
            while start < N:
                # The c-th character (1-indexed) within the current substring (which starts at `start`)
                # is located at index `start + (c - 1)` within the original string S.
                char_index_in_S = start + c - 1
                
                # According to the problem, we only consider the c-th character if the substring
                # has a length of at least c. The current substring is S[start : min(start + w, N)].
                # Its length is min(start + w, N) - start. The condition is min(start + w, N) - start >= c.
                # As derived in the thought process, since c <= w is guaranteed by the loops,
                # this length condition is equivalent to checking if the character index `start + c - 1`
                # is within the bounds of S (i.e., `start + c - 1 < N`).
                
                if char_index_in_S < N:
                    # If the index is valid (meaning the substring is long enough and the character exists),
                    # append the character at this index from S to our list.
                    result_chars.append(S[char_index_in_S])
                    
                    # Optimization: If the number of collected characters (`len(result_chars)`)
                    # exceeds the length of the target string T (`M`), then this specific pair
                    # (c, w) cannot possibly form T. We can stop processing further substrings
                    # for this (c, w) pair and move on to the next pair.
                    if len(result_chars) > M:
                        break # Break the inner while loop (over start)
                else:
                    # If the calculated character index `char_index_in_S` is already out of bounds for S,
                    # it means we have processed all possible starting positions `start` from which
                    # a valid c-th character could be extracted. Any subsequent `start` values
                    # (start + w, start + 2w, etc.) would result in even larger indices `start' + c - 1`,
                    # which would also be out of bounds. So, we can stop processing starting positions
                    # for the current (c, w) pair.
                    break # Break the inner while loop (over start)
                
                # Move to the starting index of the next substring by adding w.
                start += w
            
            # After iterating through all relevant starting positions for the current (c, w) pair,
            # we have collected all applicable characters into `result_chars`.
            # Now, check if the length of the collected characters equals the length of T,
            # and if the concatenated string of collected characters is exactly equal to T.
            if len(result_chars) == M and "".join(result_chars) == T:
                # If both conditions are met, we have found a valid pair (c, w) that satisfies
                # the problem requirements. Print "Yes" and terminate the program immediately.
                print("Yes")
                return # Exit the function

    # If the outer loops complete without finding any valid pair (c, w) that forms T,
    # it means no such pair exists under the given constraints. Print "No".
    print("No")

# Call the solve function to execute the logic.
solve()