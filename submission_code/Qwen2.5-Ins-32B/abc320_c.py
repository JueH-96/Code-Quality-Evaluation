# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.readline

    M = int(input())
    S = [input().strip() for _ in range(3)]

    # Check all possible times from 0 to 3*M-1
    for t in range(3*M):
        chars = set(S[i][(t % M)] for i in range(3))
        if len(chars) == 1:
            return t
    return -1

print(solve())