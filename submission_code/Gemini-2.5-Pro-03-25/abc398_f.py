# YOUR CODE HERE
import sys

def compute_kmp_pi(text):
    """
    Computes the KMP prefix function (pi table) for a given text.
    The prefix function pi[i] for a string text at index i is defined as the length 
    of the longest proper prefix of the substring text[0...i] which is also 
    a suffix of this substring. A proper prefix of a string is a prefix 
    that is not equal to the string itself.

    Args:
        text (str): The input string for which to compute the pi table.

    Returns:
        list[int]: The KMP prefix function (pi table). pi[i] contains the length 
                   of the longest proper prefix of text[0..i] that is also a suffix.
    """
    m = len(text)
    pi = [0] * m  # Initialize pi table with zeros
    
    # j stores the length of the previous longest prefix suffix.
    # It represents the state in the KMP automaton / the length of the current border.
    j = 0 
    
    # pi[0] is always 0, so the loop starts from i = 1
    for i in range(1, m):
        # If the current character text[i] does not match the character after the current prefix border text[j],
        # we need to find the next shorter possible prefix suffix.
        # We use the pi table itself to find the length of the next candidate prefix suffix (border).
        # This is done by setting j = pi[j-1]. We repeat this until j is 0 or we find a match.
        while j > 0 and text[i] != text[j]:
            j = pi[j-1]
        
        # If the current character text[i] matches the character text[j],
        # it means we have extended the current prefix suffix by one character.
        # So, we increment j.
        if text[i] == text[j]:
            j += 1
        
        # Store the length j of the longest prefix suffix for the prefix ending at index i.
        pi[i] = j
        
    return pi

def solve():
    """
    Reads the input string S, finds the shortest palindrome P that has S as a prefix,
    and prints P to standard output.
    """
    # Read the input string S from standard input and remove leading/trailing whitespace.
    S = sys.stdin.readline().strip()
    N = len(S)
    
    # Handle the edge case of an empty input string. The shortest palindrome is empty string itself.
    if N == 0:
        print("")
        return

    # The core idea is to find the longest suffix of S that is also a palindrome.
    # Let this suffix be S[k..N-1]. The length of this suffix is L = N - k.
    # The shortest palindrome P that starts with S can be constructed as P = S + S[0..k-1]^R.
    # S[0..k-1]^R is the reverse of the prefix S[0..k-1].
    # To find the length L of the longest palindromic suffix efficiently, we can use KMP algorithm.
    # The length L is equal to the length of the longest prefix of S^R (reverse of S) that is also a suffix of S.
    
    # Compute the reverse of S.
    S_reversed = S[::-1]
    
    # Construct a temporary string T by concatenating S_reversed, a separator character '#', and S.
    # The separator '#' must be a character that does not appear in S. Since S consists of uppercase English letters,
    # '#' is a safe choice. This separator ensures that matches do not cross the boundary between S_reversed and S incorrectly.
    T = S_reversed + "#" + S
    
    # Compute the KMP prefix function (pi table) for the constructed string T.
    pi = compute_kmp_pi(T)
    
    # The last value in the pi table, pi[-1] (or pi[len(T)-1]), gives the length of the longest proper prefix of T
    # that is also a suffix of T. Because of the structure T = S_reversed + # + S, this value corresponds to
    # the length of the longest prefix of S_reversed that is also a suffix of S.
    # This length is exactly L, the length of the longest palindromic suffix of S.
    longest_palindromic_suffix_len = pi[-1]
    
    # Calculate k, the length of the prefix of S that needs to be reversed and appended.
    # k = N - L. This is the number of characters from the beginning of S that are *not* part of the longest palindromic suffix.
    k = N - longest_palindromic_suffix_len
    
    # Extract the prefix of S of length k. This is S[0..k-1].
    prefix_to_append = S[:k]
    
    # Reverse this prefix. This gives S[0..k-1]^R.
    reversed_prefix_to_append = prefix_to_append[::-1]
    
    # Construct the final shortest palindrome P by concatenating S and the reversed prefix.
    result_palindrome = S + reversed_prefix_to_append
    
    # Print the resulting shortest palindrome to standard output.
    print(result_palindrome)

# Execute the solve function to process the input and produce the output.
solve()