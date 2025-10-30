def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N

    fixed_s = None

    # Check for fixed sums
    for i in range(N):
        a = A[i]
        b = B[i]
        if a != -1 and b != -1:
            current_sum = a + b
            if fixed_s is None:
                fixed_s = current_sum
            else:
                if current_sum != fixed_s:
                    print("No")
                    return

    # Compute s
    if fixed_s is not None:
        s = fixed_s
    else:
        lower = 0
        for i in range(N):
            a = A[i]
            b = B[i]
            if a != -1 and b == -1:
                lower = max(lower, a)
            if b != -1 and a == -1:
                lower = max(lower, b)
        s = lower

    # Check constraints
    for i in range(N):
        a = A[i]
        b = B[i]
        if a != -1 and b == -1:
            if a > s:
                print("No")
                return
        if b != -1 and a == -1:
            if b > s:
                print("No")
                return

    print("Yes")

if __name__ == "__main__":
    main()