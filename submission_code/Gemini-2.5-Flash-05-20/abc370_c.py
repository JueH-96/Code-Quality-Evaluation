import sys

def solve():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    N = len(S)
    
    # Calculate the minimum number of elements (M) required in array X.
    # This is equal to the number of positions where S and T initially differ.
    initial_diff_count = 0
    for i in range(N):
        if S[i] != T[i]:
            initial_diff_count += 1

    M = initial_diff_count
    
    # If S is already equal to T, no operations are needed, so M is 0.
    if M == 0:
        print(0)
        return

    result_X = []
    # Convert the current string S into a list of characters for efficient modification.
    # Python strings are immutable, so direct character changes are not efficient.
    current_s_chars = list(S) 

    # Perform M operations. Each operation fixes one differing character.
    for _ in range(M):
        min_candidate_s = None # Stores the lexicographically smallest string found in this step
        
        # Iterate through all possible character positions to find the best one to change.
        for i in range(N):
            # We only consider changing characters that currently differ from their target in T.
            # Changing a character that already matches T[i] would either introduce a new
            # difference or revert a previously fixed one, violating the minimum M constraint.
            if current_s_chars[i] != T[i]:
                # Create a temporary copy of the current character list to apply the change.
                temp_s_chars = list(current_s_chars) 
                # Change the character at position 'i' to its target value from T.
                temp_s_chars[i] = T[i] 
                
                # Convert the modified list of characters back into a string for comparison.
                candidate_s = "".join(temp_s_chars)

                # Compare this candidate string with the current minimum candidate.
                # If it's the first candidate, or if it's lexicographically smaller, update min_candidate_s.
                if min_candidate_s is None or candidate_s < min_candidate_s:
                    min_candidate_s = candidate_s
        
        # After checking all possible single-character changes, min_candidate_s holds
        # the lexicographically smallest string that results from one valid operation.
        result_X.append(min_candidate_s)
        
        # Update the current_s_chars for the next iteration to reflect the chosen change.
        current_s_chars = list(min_candidate_s) 

    # Print the total number of operations (M).
    print(M)
    # Print each string in the generated array X on a new line.
    for s_in_x in result_X:
        print(s_in_x)

solve()