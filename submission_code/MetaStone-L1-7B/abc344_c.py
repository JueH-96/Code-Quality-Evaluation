def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0

    # Read N and A
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    ptr += N

    # Read M and B
    M = int(input[ptr])
    ptr += 1
    B = list(map(int, input[ptr:ptr + M]))
    ptr += M

    # Read L and C
    L = int(input[ptr])
    ptr += 1
    C = list(map(int, input[ptr:ptr + L]))
    ptr += L

    # Read Q and X list
    Q = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr + Q]))

    # Precompute all possible sums
    sum_set = set()
    for a in A:
        for b in B:
            for c in C:
                sum_set.add(a + b + c)

    # Process each query
    for x in X:
        print("Yes" if x in sum_set else "No")

if __name__ == "__main__":
    main()