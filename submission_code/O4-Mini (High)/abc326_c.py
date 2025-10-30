import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    A.sort()
    ans = 0
    j = 0
    for i in range(N):
        # Move j forward while A[j] < A[i] + M
        # j never moves backwards overall, so total work is O(N)
        while j < N and A[j] < A[i] + M:
            j += 1
        # now A[j] >= A[i] + M or j == N
        # all indices in [i, j) satisfy A[k] < A[i] + M
        cnt = j - i
        if cnt > ans:
            ans = cnt

    print(ans)

if __name__ == "__main__":
    main()