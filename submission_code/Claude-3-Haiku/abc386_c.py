import sys

def solve(K, S, T):
    if len(S) == len(T):
        if S == T:
            return "Yes"
        for i in range(len(S)):
            if S[:i] + S[i+1:] == T:
                return "Yes"
        for i in range(len(S)+1):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if S[:i] + c + S[i:] == T:
                    return "Yes"
        for i in range(len(S)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if S[:i] + c + S[i+1:] == T:
                    return "Yes"
        return "No"
    elif len(S) == len(T) - 1:
        for i in range(len(S)+1):
            if S[:i] + S[i:] == T:
                return "Yes"
        return "No"
    elif len(S) == len(T) + 1:
        for i in range(len(T)+1):
            if T[:i] + T[i:] == S:
                return "Yes"
        return "No"
    else:
        return "No"

K = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

print(solve(K, S, T))