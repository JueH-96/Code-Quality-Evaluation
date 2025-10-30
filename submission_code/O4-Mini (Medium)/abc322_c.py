import sys
import threading

def main():
    import sys
    data = sys.stdin.readline().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, sys.stdin.readline().split()))
    # Two pointers: for each day i, find the earliest A[j] >= i
    res = [0] * N
    j = 0
    for i in range(1, N+1):
        # advance j until A[j] >= i
        while j < M and A[j] < i:
            j += 1
        # A[j] is the first fireworks day >= i
        res[i-1] = A[j] - i
    # output
    out = "
".join(map(str, res))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()