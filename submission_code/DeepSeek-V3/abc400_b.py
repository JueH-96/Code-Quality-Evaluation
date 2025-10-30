S = input().strip()
n = len(S)
if n == 1:
    print(S)
else:
    # Function to compute the longest palindromic prefix using KMP's lps array
    def longest_palindromic_prefix(s):
        s_rev = s[::-1]
        combined = s + '#' + s_rev
        lps = [0] * len(combined)
        for i in range(1, len(combined)):
            j = lps[i-1]
            while j > 0 and combined[i] != combined[j]:
                j = lps[j-1]
            if combined[i] == combined[j]:
                j += 1
            lps[i] = j
        return lps[-1]
    
    max_len = longest_palindromic_prefix(S)
    if max_len == n:
        print(S)
    else:
        remaining = S[max_len:]
        palindrome = S + remaining[::-1]
        print(palindrome)