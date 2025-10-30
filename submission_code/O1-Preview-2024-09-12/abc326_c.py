# YOUR CODE HERE
import sys
import threading
def main():
    import bisect
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    maxcount = 0
    N = len(A)
    j = 0
    for i in range(N):
        while j < N and A[j] < A[i] + M:
            j +=1
        count = j - i
        maxcount = max(maxcount, count)
    print(maxcount)
threading.Thread(target=main).start()