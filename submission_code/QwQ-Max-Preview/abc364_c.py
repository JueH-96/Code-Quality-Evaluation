def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    Y = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N

    # Sort A and B in descending order
    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B, reverse=True)

    # Compute prefix sums
    a_prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        a_prefix[i] = a_prefix[i-1] + A_sorted[i-1]
    b_prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        b_prefix[i] = b_prefix[i-1] + B_sorted[i-1]

    min_ans = float('inf')
    for m in range(1, N + 1):
        if a_prefix[m] > X or b_prefix[m] > Y:
            if m < min_ans:
                min_ans = m
    if min_ans == float('inf'):
        print(N)
    else:
        print(min_ans)

if __name__ == "__main__":
    main()