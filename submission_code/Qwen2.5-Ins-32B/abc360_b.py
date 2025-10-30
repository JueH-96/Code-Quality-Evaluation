import sys

def solve(S, T):
    for w in range(1, len(S)):
        for c in range(1, w + 1):
            result = ''
            for i in range(c - 1, len(S), w):
                if i + 1 <= len(S):
                    result += S[i]
            if result == T:
                return 'Yes'
    return 'No'

S, T = input().strip().split()
print(solve(S, T))