import sys

# Function to check if two strings are equal or differ by one character
# Assumes len(s1) == len(s2)
def is_one_change_or_equal(s1, s2):
    diff_count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            diff_count += 1
        # Optimization: if diff_count exceeds 1, no need to continue
        if diff_count > 1:
            return False
    # True if diff_count is 0 (equal) or 1 (one change)
    return diff_count <= 1

# Function to check if s2 can be obtained by deleting one character from s1
# Assumes len(s1) == len(s2) + 1
def is_one_delete(s1, s2):
    p1 = 0 # pointer for s1
    p2 = 0 # pointer for s2
    
    # Match initial common prefix
    while p1 < len(s1) and p2 < len(s2) and s1[p1] == s2[p2]:
        p1 += 1
        p2 += 1
        
    # After the first matching loop, either we reached the end of s2 (s2 was a prefix),
    # or we found the first mismatch at s1[p1] and s2[p2].

    # If p2 reached end of s2, s2 is s1 without its last character.
    # This happens when s1 = s2 + s1[-1]
    if p2 == len(s2):
         return True
    
    # If mismatch found (p2 < len(s2)), the character s1[p1] must be the potentially deleted one.
    # Skip s1[p1] and match the rest of s1 with the rest of s2.
    # p1 is currently at the index of the mismatch in s1. Increment p1 to skip it.
    p1 += 1 
    
    # Match remaining parts starting from the character after the potential deletion in s1.
    while p1 < len(s1) and p2 < len(s2) and s1[p1] == s2[p2]:
        p1 += 1
        p2 += 1
        
    # If p2 reached the end of s2, it means all characters in s2 were matched against
    # characters in s1, accounting for exactly one skipped character in s1.
    return p2 == len(s2)


def main():
    # Read N and T' from the first line
    # Use sys.stdin.readline for potentially large inputs
    input_line = sys.stdin.readline().split()
    N = int(input_line[0])
    T_prime = input_line[1].strip() # Ensure no trailing newline
    
    candidate_indices = []
    
    len_T_prime = len(T_prime)
    
    # Iterate through the N strings S_i
    for i in range(1, N + 1):
        S_i = sys.stdin.readline().strip() # Read each S_i
        len_S_i = len(S_i)
        
        # Calculate length difference
        len_diff = len_S_i - len_T_prime
        
        # Check if the length difference is allowed
        # If |len(S_i) - len(T')| > 1, S_i cannot be T under any of the four conditions
        if abs(len_diff) > 1:
            continue
            
        is_candidate = False
        
        if len_diff == 0:
            # Case: |S_i| == |T_prime|. Corresponds to T'=T or T' is T with one change.
            # Check if S_i is equal to T_prime or differs by one character.
            is_candidate = is_one_change_or_equal(S_i, T_prime)
            
        elif len_diff == -1: # len_S_i == len_T_prime - 1
            # Case: |S_i| == |T_prime| - 1. Corresponds to T' is T with one insertion.
            # This means T (S_i) is T' with one deletion.
            # Check if S_i can be obtained by deleting one character from T_prime.
            # Call is_one_delete(T_prime, S_i)
            is_candidate = is_one_delete(T_prime, S_i)
            
        elif len_diff == 1: # len_S_i == len_T_prime + 1
            # Case: |S_i| == |T_prime| + 1. Corresponds to T' is T with one deletion.
            # This means T (S_i) is T' with one insertion.
            # Equivalently, check if T_prime can be obtained by deleting one character from S_i.
            # Call is_one_delete(S_i, T_prime)
            is_candidate = is_one_delete(S_i, T_prime)
            
        if is_candidate:
            candidate_indices.append(i)
            
    # Print the number of candidates
    print(len(candidate_indices))
    
    # Print the indices of the candidates, separated by spaces
    if candidate_indices:
        print(*candidate_indices)

if __name__ == "__main__":
    main()