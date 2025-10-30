import sys

def solve():
    S_str = sys.stdin.readline().strip()
    T_str = sys.stdin.readline().strip()
    N = len(S_str)

    if S_str == T_str:
        print(0)
        return

    # Current state of S, as a list of characters for easy modification
    s_chars = list(S_str) 

    # X will store the sequence of strings generated
    X = [] 

    # Loop until current S (represented by s_chars) equals T_str
    while "".join(s_chars) != T_str:
        # Find all indices where s_chars differs from T_str.
        # These are the positions where a change can be made in this step.
        diff_indices = []
        for i in range(N):
            if s_chars[i] != T_str[i]:
                diff_indices.append(i)
        
        # Generate all possible strings that can be formed by changing one character
        # in s_chars to match the corresponding character in T_str.
        candidate_next_strings = []
        for idx_to_change in diff_indices:
            # Create a temporary copy of s_chars to simulate the change
            temp_s_chars = list(s_chars) 
            temp_s_chars[idx_to_change] = T_str[idx_to_change] # Apply the change
            candidate_next_strings.append("".join(temp_s_chars))
        
        # From all candidate strings, choose the one that is lexicographically smallest.
        # This greedy choice ensures X is lexicographically smallest overall.
        best_next_s_str = min(candidate_next_strings)
        
        # Update s_chars to reflect this chosen change for the next iteration
        s_chars = list(best_next_s_str)
        
        # Append the resulting string (S after the change) to X
        X.append(best_next_s_str)

    # Print the number of operations (which is the length of X)
    print(len(X))
    # Print each string in X, one per line
    for s_in_x in X:
        print(s_in_x)

if __name__ == '__main__':
    solve()