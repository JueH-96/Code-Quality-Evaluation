# YOUR CODE HERE
import sys
import threading

def main():
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    N = len(S)
    acc = [0] * N
    for i in range(1, N):
        acc[i] = acc[i - 1] + (1 if S[i] == S[i - 1] else 0)
    for _ in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        l -= 1  # adjust to 0-based index
        r -= 1
        ans = acc[r] - acc[l]
        print(ans)
        
threading.Thread(target=main).start()