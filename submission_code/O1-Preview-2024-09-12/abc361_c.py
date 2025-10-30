# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    N_minus_K = N - K
    min_diff = float('inf')
    for i in range(K+1):
        start = i
        end = i + N_minus_K -1
        diff = A[end] - A[start]
        if diff < min_diff:
            min_diff = diff
    print(min_diff)
threading.Thread(target=main).start()