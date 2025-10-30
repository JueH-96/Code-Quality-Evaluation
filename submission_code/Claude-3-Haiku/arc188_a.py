from collections import defaultdict

def count_good_strings(N, K, S):
    mod = 998244353
    
    # Count the occurrences of A, B, and C in the string
    count = defaultdict(int)
    for c in S:
        if c != '?':
            count[c] += 1
    
    # If there are not enough A, B, and C, return 0
    if min(count['A'], count['B'], count['C']) < 1:
        return 0
    
    # Compute the number of good substrings
    good_substrings = 0
    for i in range(N):
        for j in range(i, N):
            substring = S[i:j+1]
            if is_good_string(substring):
                good_substrings += 1
                if good_substrings >= K:
                    break
    
    # If we have at least K good substrings, return 1
    if good_substrings >= K:
        return 1
    
    # Otherwise, try to replace the '?' characters
    return replace_question_marks(N, K, S, count)

def is_good_string(s):
    # Check if the string can be turned into an empty string using the given operations
    while True:
        # Try to perform Operation 1
        found = False
        for i in range(len(s)-1):
            if s[i] == s[i+1]:
                s = s[:i] + s[i+2:]
                found = True
                break
        if found:
            continue
        
        # Try to perform Operation 2
        a, b, c = False, False, False
        for c in s:
            if c == 'A':
                a = True
            elif c == 'B':
                b = True
            elif c == 'C':
                c = True
        if a and b and c:
            s = s.replace('A', '', 1)
            s = s.replace('B', '', 1)
            s = s.replace('C', '', 1)
            continue
        
        # If neither operation can be performed, the string is good
        return len(s) == 0

def replace_question_marks(N, K, S, count):
    mod = 998244353
    
    # Try all possible replacements of '?' with A, B, or C
    total = 0
    for a in range(count['?']+1):
        for b in range(count['?']+1-a):
            c = count['?'] - a - b
            new_S = S.replace('?', 'A', a).replace('?', 'B', b).replace('?', 'C', c)
            total += count_good_strings(N, K, new_S)
            total %= mod
    
    return total

# Read the input
N, K = map(int, input().split())
S = input()

# Solve the problem and print the answer
print(count_good_strings(N, K, S))