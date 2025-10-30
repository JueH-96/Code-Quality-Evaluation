import bisect

def main():
    n = int(input().strip())
    A = list(map(int, input().split()))
    total = 0
    for j in range(n):
        threshold = A[j] // 2
        count_j = bisect.bisect_right(A, threshold, 0, j)
        total += count_j
    print(total)

if __name__ == "__main__":
    main()