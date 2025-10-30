# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    positions = [0] * (N + 1)
    for idx, val in enumerate(P):
        positions[val] = idx + 1  # positions from 1 to N

    ans = float('inf')
    for a in range(1, N - K + 2):
        indices = [positions[x] for x in range(a, a + K)]
        indices.sort()
        i1 = indices[0]
        iK = indices[K - 1]
        ans = min(ans, iK - i1)
    print(ans)

threading.Thread(target=main).start()