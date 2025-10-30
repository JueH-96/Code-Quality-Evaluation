import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    S_str = sys.stdin.readline().strip()
    S_list = list(S_str)

    # Helper function to check if "ABC" starts at index idx
    # length_N is the total length of the string S_list
    def is_abc_at(s_l, idx, length_N):
        # Check if idx, idx+1, idx+2 are all valid indices for a 3-character substring
        # Valid starting indices for a 3-char substring are 0 to length_N - 3
        if not (0 <= idx <= length_N - 3):
            return False
        # Check if the characters form "ABC"
        return s_l[idx] == 'A' and s_l[idx+1] == 'B' and s_l[idx+2] == 'C'

    # Initial count of "ABC"
    current_abc_count = 0
    # Iterate through all possible starting positions for "ABC"
    # The last possible start index for "ABC" is N-3.
    # range(N-2) goes from 0 up to N-3.
    for i in range(N - 2): 
        if is_abc_at(S_list, i, N):
            current_abc_count += 1
    
    # Process queries
    for _ in range(Q):
        X_str, C_char = sys.stdin.readline().split()
        X = int(X_str)
        k = X - 1 # Convert to 0-indexed

        # If the character doesn't actually change, the count remains the same
        if S_list[k] == C_char:
            sys.stdout.write(str(current_abc_count) + "
")
            continue
        
        # The character at S_list[k] is about to change.
        # The set of starting indices of 3-char substrings that could be affected is {k-2, k-1, k}.
        affected_start_indices = {k-2, k-1, k}

        # Before changing S_list[k]:
        # For each potentially affected "ABC" pattern, if it exists, decrement count.
        # This check uses S_list with the character at S_list[k] still being the OLD char.
        for start_idx in affected_start_indices:
            if is_abc_at(S_list, start_idx, N): 
                current_abc_count -= 1
        
        # Perform the character update
        S_list[k] = C_char

        # After changing S_list[k]:
        # For each potentially affected "ABC" pattern, if it now exists, increment count.
        # This check uses S_list with the character at S_list[k] now being the NEW char.
        for start_idx in affected_start_indices:
            if is_abc_at(S_list, start_idx, N): 
                current_abc_count += 1
        
        sys.stdout.write(str(current_abc_count) + "
")

if __name__ == '__main__':
    main()