import bisect
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx + N]))
    A.sort()
    max_count = 0
    for i in range(N):
        target = A[i] + M
        j = bisect.bisect_left(A, target, i, N)
        count = j - i
        if count > max_count:
            max_count = count
    print(max_count)

if __name__ == '__main__':
    main()