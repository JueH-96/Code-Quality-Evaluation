# YOUR CODE HERE
def can_form_t(S, T):
    len_s = len(S)
    len_t = len(T)
    
    for w in range(1, len_s):
        for c in range(1, w + 1):
            formed_t = []
            for i in range(0, len_s, w):
                if i + c <= len_s:
                    formed_t.append(S[i + c - 1])
            if ''.join(formed_t) == T:
                return "Yes"
    return "No"

import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
T = data[1]

print(can_form_t(S, T))