import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        if query[0] == 1:
            for i in range(query[1]-1, query[2]):
                S = S[:i] + '0' + S[i+1:]
        elif query[0] == 2:
            S_prime = S[query[1]-1:query[2]]
            if is_good_string(S_prime):
                print('Yes')
            else:
                print('No')

def is_good_string(s):
    for i in range(len(s)-1):
        if s[i] == s[i+1]:
            return False
    return True

if __name__ == "__main__":
    solve()