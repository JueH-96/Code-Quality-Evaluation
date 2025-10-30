import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    A.sort()
    total = 0
    M = A[-1]
    for i in range(n):
        k = A[i]
        if i == n - 1:
            continue
        current_max_q = M // k
        current_sum = 0
        for q in range(1, current_max_q + 1):
            val = q * k
            pos = bisect.bisect_left(A, val, i + 1, n)
            current_sum += (n - pos)
        total += current_sum
    print(total)

if __name__ == "__main__":
    main()