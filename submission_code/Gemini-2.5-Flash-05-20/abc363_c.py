# YOUR CODE HERE
import itertools

def is_palindrome(s):
    """
    Checks if a string is a palindrome.
    A string is a palindrome if it reads the same forwards and backwards.
    """
    return s == s[::-1]

def solve():
    # Read N and K from the first line of input, splitting by space and converting to integers.
    N, K = map(int, input().split())
    # Read the string S from the second line of input.
    S = input()

    # Generate all unique permutations of the string S.
    # itertools.permutations(S) yields tuples of characters (e.g., ('a', 'a', 'b')).
    # We join these characters into a single string (e.g., "aab").
    # Storing them in a set automatically handles uniqueness, so if S has duplicate characters,
    # we only process each distinct permutation once.
    unique_permutations = set()
    for p_tuple in itertools.permutations(S):
        unique_permutations.add("".join(p_tuple))

    count = 0
    # Iterate through each unique permuted string.
    for p_str in unique_permutations:
        # This flag will be set to True if we find any palindrome of length K
        # as a substring within the current permutation.
        has_palindrome_k_substring = False
        
        # Check all possible substrings of length K.
        # A substring of length K starting at index `i` will extend to `i + K - 1`.
        # The last possible starting index `i` is `N - K`.
        # So, we iterate `i` from 0 up to `N - K` (inclusive).
        for i in range(N - K + 1):
            substring = p_str[i : i + K] # Extract the substring of length K
            
            # Check if the extracted substring is a palindrome using our helper function.
            if is_palindrome(substring):
                has_palindrome_k_substring = True
                # If we found a palindrome, no need to check further substrings for this `p_str`.
                # We can break out of the inner loop and move to the next permutation.
                break 
        
        # After checking all substrings (or breaking early), if `has_palindrome_k_substring`
        # is still False, it means this permuted string does NOT contain any palindrome
        # of length K. So, we count it.
        if not has_palindrome_k_substring:
            count += 1

    # Print the final count of strings that satisfy the condition.
    print(count)

# Call the solve function to execute the program.
solve()