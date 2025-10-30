# YOUR CODE HERE
def can_form_string(S, T):
    for w in range(1, len(S)):
        for c in range(1, w + 1):
            formed = []
            for i in range(0, len(S), w):
                substring = S[i:i + w]
                if len(substring) >= c:
                    formed.append(substring[c - 1])
            if ''.join(formed) == T:
                return "Yes"
    return "No"

import sys
input = sys.stdin.read
data = input().split()
S = data[0]
T = data[1]

print(can_form_string(S, T))