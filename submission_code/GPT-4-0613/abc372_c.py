import sys
from collections import deque

def main():
    N, Q = map(int, sys.stdin.readline().split())
    S = list(sys.stdin.readline().strip())
    queries = [list(sys.stdin.readline().split()) for _ in range(Q)]
    queries = [(int(x[0])-1, x[1]) for x in queries]

    ABC = deque([0]*(N+1), maxlen=N+1)
    for i in range(N-2):
        if S[i:i+3] == list('ABC'):
            ABC[i+3] = ABC[i+2] + 1
        else:
            ABC[i+3] = ABC[i+2]

    for x, c in queries:
        old = S[x]
        S[x] = c
        if x >= 2 and S[x-2:x+1] == list('ABC'):
            ABC[x+1] = ABC[x-2] + 1
        elif x >= 1 and S[x-1:x+2] == list('ABC'):
            ABC[x+2] = ABC[x-1] + 1
        elif S[x:x+3] == list('ABC'):
            ABC[x+3] = ABC[x] + 1
        else:
            ABC[x+1] = ABC[x]
            if x < N-1:
                ABC[x+2] = ABC[x+1]
            if x < N-2:
                ABC[x+3] = ABC[x+2]
        print(ABC[-1])

if __name__ == "__main__":
    main()