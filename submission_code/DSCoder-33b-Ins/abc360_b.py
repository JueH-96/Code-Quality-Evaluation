# YOUR CODE HERE

S, T = input().split()

def solve(S, T):
    for w in range(1, len(S)):
        for c in range(1, w+1):
            if ''.join(S[i:i+c] for i in range(0, w, c)) == T:
                return 'Yes'
    return 'No'

print(solve(S, T))