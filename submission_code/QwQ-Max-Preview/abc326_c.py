import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+N]))
    A.sort()
    max_count = 0
    for i in range(N):
        target = A[i] + M
        j = bisect.bisect_left(A, target) - 1
        current_count = j - i + 1
        if current_count > max_count:
            max_count = current_count
    print(max_count)

if __name__ == "__main__":
    main()