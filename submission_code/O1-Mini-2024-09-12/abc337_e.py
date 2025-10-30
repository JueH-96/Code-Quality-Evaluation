import math
import sys

def solve():
    import sys, math
    N = int(sys.stdin.readline())
    if N == 1:
        M = 1
    else:
        M = math.ceil(math.log2(N))
    # Assign bottles to friends
    friend_bottles = [[] for _ in range(M)]
    bottle_patterns = {}
    for i in range(1, N+1):
        bits = bin(i-1)[2:].zfill(M)[::-1]
        pattern = ''.join(bits)
        bottle_patterns[pattern] = i
        for j in range(M):
            if bits[j] == '1':
                friend_bottles[j].append(i)
    # Output M
    print(M)
    sys.stdout.flush()
    # Output M lines for each friend
    for j in range(M):
        bottles = sorted(friend_bottles[j])
        K = len(bottles)
        if K == 0:
            print("0")
        else:
            print(f"{K} " + " ".join(map(str, bottles)))
        sys.stdout.flush()
    # Read S
    S = sys.stdin.readline().strip()
    # Find X'
    if len(S) != M:
        X = 1
    else:
        X = bottle_patterns.get(S, 1)
    print(X)
    sys.stdout.flush()