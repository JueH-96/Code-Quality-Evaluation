import sys
import threading

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(M)]
    sums = [0] * M
    for _ in range(N):
        for j in range(M):
            sums[j] += int(next(it))
    for j in range(M):
        if sums[j] < A[j]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()