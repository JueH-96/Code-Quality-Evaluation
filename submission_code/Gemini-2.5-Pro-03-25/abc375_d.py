# YOUR CODE HERE
import sys

# Function to map 'A'.. 'Z' to 0..25
def char_to_int(c):
    """Converts an uppercase English letter to an integer 0-25."""
    return ord(c) - ord('A')

def solve():
    """Reads input string S, computes the number of palindromic triples (i, j, k) 
    such that 1 <= i < j < k <= |S| and S_i S_j S_k is a palindrome, 
    and prints the result to standard output.
    """
    # Read input string from standard input
    S = sys.stdin.readline().strip()
    N = len(S)
    
    # A palindrome of length 3 requires S_i == S_k. The indices must satisfy 1 <= i < j < k <= |S|.
    # This implies the string must have length at least 3.
    if N < 3:
        print(0)
        return

    # We will use 0-based indexing internally for convenience.
    # The condition becomes 0 <= i < j < k <= N-1 and S[i] == S[k].
    
    # prefix_counts[p][char_code] stores the count of character corresponding to `char_code`
    # in the prefix S[0...p-1]. The array size is (N+1) x 26.
    # prefix_counts[0] represents the empty prefix, so all counts are 0.
    prefix_counts = [[0] * 26 for _ in range(N + 1)]

    # Compute prefix counts efficiently
    for i in range(N):
        # Get the integer code (0-25) for the character S[i]
        char_code = char_to_int(S[i])
        
        # Initialize counts for prefix ending at index i (length i+1) based on previous prefix counts
        for k in range(26):
            prefix_counts[i+1][k] = prefix_counts[i][k]
            
        # Increment the count for the specific character S[i] found at index i
        prefix_counts[i+1][char_code] += 1

    # total_counts[char_code] stores the total count of the character in the entire string S.
    # This is available from the counts of the full prefix S[0...N-1], stored in prefix_counts[N].
    total_counts = prefix_counts[N]
    
    total_palindromes = 0
    
    # Iterate through all possible middle indices j (using 0-based indexing).
    # The condition for triples is 0 <= i < j < k <= N-1.
    # This implies that the middle index j must be strictly between the first and last possible indices.
    # So, j must satisfy 1 <= j <= N-2.
    for j in range(1, N - 1):
        # For a fixed middle index j, we need to find pairs (i, k) such that
        # 0 <= i < j (i is to the left of j)
        # and j < k <= N-1 (k is to the right of j)
        # and S[i] == S[k] (the characters must match for the triple S[i]S[j]S[k] to be a palindrome).
        
        # We can count these pairs efficiently by iterating through each possible character C ('A' through 'Z').
        for char_code in range(26):
            # Count occurrences of character C to the left of index j.
            # These are indices i such that 0 <= i < j.
            # The count of C in the prefix S[0...j-1] is stored in prefix_counts[j][char_code].
            count_left = prefix_counts[j][char_code] 
            
            # If there are no occurrences of C to the left, this character cannot contribute to palindromes
            # with middle index j. Skip calculation for efficiency.
            if count_left == 0:
                continue

            # Count occurrences of character C to the right of index j.
            # These are indices k such that j < k <= N-1.
            # This count can be computed as: Total count of C in S - Count of C in S[0...j].
            # The count of C in S[0...j] is stored in prefix_counts[j+1][char_code].
            count_right = total_counts[char_code] - prefix_counts[j+1][char_code]
            
            # If there are no occurrences of C to the right, this character also cannot contribute.
            if count_right == 0:
                continue

            # For the current character C, any choice of index i from the `count_left` possibilities
            # and any choice of index k from the `count_right` possibilities forms a valid pair (i, k)
            # such that S[i] = S[k] = C, with i < j < k.
            # The total number of such pairs for character C and middle index j is count_left * count_right.
            # Each such pair (i,k), combined with the middle index j, forms a valid palindromic triple (i, j, k).
            total_palindromes += count_left * count_right
            
    # Print the final total count of palindromic triples found.
    print(total_palindromes)

# Call the solve function when the script is executed.
if __name__ == '__main__':
    solve()