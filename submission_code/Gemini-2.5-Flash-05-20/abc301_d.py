import sys

def solve():
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())

    L = len(S)
    
    # ans_chars will store the constructed binary string for X.
    # It is initialized with the characters from S.
    # '?' characters will be replaced by '0' or '1' during the process.
    ans_chars = list(S)

    # This loop iterates from the most significant bit (left) to the least significant bit (right).
    # The goal is to determine '0' or '1' for each '?' character in S,
    # prioritizing '1' to maximize the resulting number, while ensuring it stays <= N.
    for i in range(L):
        # If the current character is '?', we have a choice.
        if ans_chars[i] == '?':
            # Strategy: Try to set this bit to '1' to make the number as large as possible.
            ans_chars[i] = '1'
            
            # Calculate the *smallest* possible value (current_val_if_one) that can be formed
            # if we commit to '1' at ans_chars[i]. This "smallest" value is achieved by
            # setting all subsequent '?' characters (at indices j > i) to '0'.
            current_val_if_one = 0
            for j in range(L):
                if ans_chars[j] == '1':
                    # If the character is '1', set the corresponding bit in current_val_if_one.
                    # L - 1 - j converts the left-to-right index j into a right-to-left bit position.
                    current_val_if_one |= (1 << (L - 1 - j))
                # If ans_chars[j] is '0', it contributes 0 to the value, so no action needed.
                # If ans_chars[j] is '?' and j > i, it is treated as '0' for this minimum value check,
                # so no action needed either.
                # Note: ans_chars[j] == '?' for j < i is not possible because those would have been resolved in previous iterations.
            
            # Now, check if this `current_val_if_one` (the smallest possible value if we set ans_chars[i] to '1')
            # already exceeds N.
            if current_val_if_one > N:
                # If it exceeds N, it means setting ans_chars[i] to '1' would make the number
                # too large, even if we minimize the remaining part by setting subsequent '?'s to '0'.
                # Therefore, we *must* set ans_chars[i] to '0'.
                ans_chars[i] = '0'
            # Else (current_val_if_one <= N):
            # Setting ans_chars[i] to '1' is a valid option and helps maximize the number.
            # So, we keep ans_chars[i] as '1' (it's already set to '1' from the tentative assignment).
    
    # After the loop, ans_chars now contains only '0's and '1's.
    # All original '?' characters have been definitively resolved.
    
    # Convert the final ans_chars (which is now a list representing a binary string) to an integer.
    final_val = 0
    for i in range(L):
        if ans_chars[i] == '1':
            final_val |= (1 << (L - 1 - i))
            
    # Check if the constructed number final_val is less than or equal to N.
    if final_val <= N:
        print(final_val)
    else:
        # If final_val > N, it means even with our greedy strategy trying to stay below N,
        # the resulting number still turned out to be too large.
        # This implies that no number that can be formed from S (by replacing '?')
        # is less than or equal to N.
        print(-1)

solve()