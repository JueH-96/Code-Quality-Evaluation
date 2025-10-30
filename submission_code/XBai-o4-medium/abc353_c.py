import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    mod = 10**8
    sum_a = sum(A)
    sorted_A = sorted(A)
    c = 0
    for i in range(n):
        if i == n - 1:
            break
        current = sorted_A[i]
        target = mod - current
        pos = bisect.bisect_left(sorted_A, target, i + 1, n)
        c += (n - pos)
    total = (n - 1) * sum_a - mod * c
    print(total)

if __name__ == "__main__":
    main()