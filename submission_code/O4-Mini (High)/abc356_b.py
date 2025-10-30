def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(m)]
    sums = [0] * m
    for _ in range(n):
        for j in range(m):
            sums[j] += int(next(it))
    for j in range(m):
        if sums[j] < A[j]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()