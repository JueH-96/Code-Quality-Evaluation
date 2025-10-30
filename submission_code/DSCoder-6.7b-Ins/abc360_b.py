import sys

def solve():
    S, T = sys.stdin.read().split()
    len_s = len(S)
    len_t = len(T)

    for w in range(1, len_s):
        pieces = [S[i:i+w] for i in range(0, len_s, w)]
        if ''.join([piece[0] for piece in pieces]) == T[:len_t]:
            print('Yes')
            return
    print('No')

solve()