import sys
import random

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, Q = int(input[ptr]), int(input[ptr+1])
    ptr += 2

    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Convert to 1-based indexing
    A = [0] + A
    B = [0] + B

    # Precompute random values for each possible element (1-based up to N)
    random.seed(42)
    r1 = [0] * (N + 1)
    r2 = [0] * (N + 1)
    for x in range(1, N+1):
        r1[x] = random.getrandbits(64)
        r2[x] = random.getrandbits(64)

    # Compute prefix hashes for A and B
    hA1 = [0] * (N + 1)
    hA2 = [0] * (N + 1)
    hB1 = [0] * (N + 1)
    hB2 = [0] * (N + 1)

    for i in range(1, N+1):
        a = A[i]
        hA1[i] = hA1[i-1] + r1[a]
        hA2[i] = hA2[i-1] + r2[a]

        b = B[i]
        hB1[i] = hB1[i-1] + r1[b]
        hB2[i] = hB2[i-1] + r2[b]

    # Process each query
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr+1])
        L = int(input[ptr+2])
        R = int(input[ptr+3])
        ptr += 4

        lenA = r - l + 1
        lenB = R - L + 1
        if lenA != lenB:
            print("No")
            continue

        deltaA1 = hA1[r] - hA1[l-1]
        deltaA2 = hA2[r] - hA2[l-1]
        deltaB1 = hB1[R] - hB1[L-1]
        deltaB2 = hB2[R] - hB2[L-1]

        if deltaA1 == deltaB1 and deltaA2 == deltaB2:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()