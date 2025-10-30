def can_form_t(S, T):
    len_s = len(S)
    len_t = len(T)

    for w in range(1, len_s):  # w must be less than |S|
        for c in range(1, w + 1):  # c must be between 1 and w
            concatenated = []
            for start in range(0, len_s, w):
                substring = S[start:start + w]
                if len(substring) >= c:
                    concatenated.append(substring[c - 1])  # c is 1-indexed
            if ''.join(concatenated) == T:
                return "Yes"
    
    return "No"

import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
T = data[1]

print(can_form_t(S, T))