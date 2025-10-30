import sys
from collections import deque

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    lost_socks = set(A)
    remaining_socks = [i for i in range(1, N + 1) if i not in lost_socks]

    remaining_socks = deque(remaining_socks)
    total_weirdness = 0

    while len(remaining_socks) > 1:
        sock1 = remaining_socks.popleft()
        sock2 = remaining_socks.popleft()
        total_weirdness += abs(sock1 - sock2)

    print(total_weirdness)

solve()