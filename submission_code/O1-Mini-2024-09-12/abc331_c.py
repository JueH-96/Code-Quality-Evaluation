import sys
import bisect

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    sorted_A = sorted(A)
    sum_suffix = [0] * (N + 1)
    for i in range(N-1, -1, -1):
        sum_suffix[i] = sorted_A[i] + sum_suffix[i+1]
    result = []
    for num in A:
        idx = bisect.bisect_right(sorted_A, num)
        result.append(str(sum_suffix[idx]))
    print(' '.join(result))

if __name__ == "__main__":
    main()