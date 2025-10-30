def compute_prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i-1]
        while j > 0 and s[i] != s[j]:
            j = pi[j-1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def main():
    import sys
    S = sys.stdin.read().strip()
    if not S:
        return
    
    rev_S = S[::-1]
    # Build the combined string: rev_S + '#' + S
    combined = rev_S + '#' + S
    pi = compute_prefix_function(combined)
    
    # The value at the end of pi gives length of longest palindromic suffix in S.
    L = pi[-1]
    # Append the reverse of the prefix in S that is not a palindrome.
    to_append = S[:len(S)-L][::-1]
    ans = S + to_append
    sys.stdout.write(ans)
    
if __name__ == '__main__':
    main()