# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math

    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    D = []
    K = len(A)
    for i in range(K - 1):
        D.append(A[i + 1] - A[i])
    D.sort()
    num_pairs = K // 2
    total_weirdness = sum(D[:num_pairs])
    print(total_weirdness)

threading.Thread(target=main).start()