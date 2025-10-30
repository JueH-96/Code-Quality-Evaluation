MOD = 10**9 + 7

def main():
    s = input().strip()
    t = input().strip()
    n = len(s)
    k = int(input())
    
    if len(s) != len(t):
        print(0)
        return
    
    s_concat = s + s
    t_len = len(t)
    
    # Find all occurrences of t in s_concat
    # Using KMP algorithm for efficient substring search
    def kmp_search(text, pattern):
        n = len(text)
        m = len(pattern)
        if m == 0:
            return []
        lps = [0] * m
        length = 0
        i = 1
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        i = 0  # index for text
        j = 0  # index for pattern
        indices = []
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
                if j == m:
                    indices.append(i - j)
                    j = lps[j - 1]
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        return indices
    
    occurrences = kmp_search(s_concat, t)
    valid_R = set()
    for i in occurrences:
        if i < n:
            valid_R.add(i)
    valid_R = list(valid_R)
    
    total = 0
    n_mod = n * MOD
    
    for R in valid_R:
        if R == 0:
            B = pow(-1, k, MOD) * (n - 1)
        else:
            B = pow(-1, k, MOD) * (-1)
        
        A = pow(n-1, k, n_mod)
        
        numerator = (A + B) % n_mod
        if numerator < 0:
            numerator += n_mod
        
        ways = numerator // n
        total = (total + ways) % MOD
    
    print(total % MOD)

if __name__ == "__main__":
    main()