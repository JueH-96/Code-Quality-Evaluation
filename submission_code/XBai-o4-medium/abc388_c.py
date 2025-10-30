import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    total = 0
    for j in range(n):
        target = A[j] / 2
        cnt = bisect.bisect_right(A, target, 0, j)
        total += cnt
    print(total)

if __name__ == "__main__":
    main()