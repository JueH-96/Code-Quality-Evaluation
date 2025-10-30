import sys

def solve():
    S = sys.stdin.readline().strip()
    N = len(S)

    # If S is already a palindrome, the longest palindromic suffix is S itself (length N).
    # The algorithm handles this case correctly, resulting in an empty string to append.

    S_rev = S[::-1] # Reverse of S

    # Construct the string for KMP.
    # The pattern is S_rev, the text part is S. '#' is a delimiter.
    # We are looking for the longest prefix of S_rev that matches a suffix of S.
    # If S_rev[0...L-1] == S[N-L...N-1], then S[N-L...N-1] is a palindrome.
    # This is because S_rev[0...L-1] is the reverse of S[0...L-1].
    # So, S[0...L-1][::-1] == S[N-L...N-1], which means the suffix of S is a palindrome.
    text = S_rev + '#' + S
    
    lps = [0] * len(text) # Longest Proper Prefix which is also a Suffix array
    length = 0 # Length of the previous longest prefix suffix
    
    # KMP LPS array computation
    for i in range(1, len(text)):
        while length > 0 and text[i] != text[length]:
            length = lps[length - 1]
        
        if text[i] == text[length]:
            length += 1
        
        lps[i] = length
            
    # The last element of the LPS array (lps[len(text) - 1])
    # stores the length of the longest prefix of `text` that is also a suffix of `text`.
    # This length corresponds to the length of the longest palindromic suffix of S.
    matched_len = lps[len(text) - 1]
    
    # The part of S that is not part of the longest palindromic suffix needs to be reversed
    # and appended to S. This part is S[0 : N - matched_len].
    
    suffix_to_add = S[0 : N - matched_len][::-1]
    
    result = S + suffix_to_add
    
    sys.stdout.write(result + "
")

# Call the solve function to execute the logic
solve()