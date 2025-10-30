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
    # two pointers: for each i, advance j while A[j] < A[i] + M
    for i in range(N):
        # ensure j is at least i
        if j < i:
            j = i
        # move j forward as long as the interval [A[i], A[i]+M) includes A[j]
        while j < N and A[j] < A[i] + M:
            j += 1
        # now A[j] is the first outside, so count is j - i
        cnt = j - i
        if cnt > ans:
            ans = cnt

    print(ans)

if __name__ == "__main__":
    main()