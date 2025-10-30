import sys
import random

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Generate hash values for each possible number (1 to N)
    max_num = N
    h1 = [0] * (max_num + 1)
    h2 = [0] * (max_num + 1)
    for x in range(1, max_num + 1):
        h1[x] = random.getrandbits(64)
        h2[x] = random.getrandbits(64)

    # Precompute prefix sums for A and B
    prefix1_A = [0] * (N + 1)
    prefix2_A = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix1_A[i] = prefix1_A[i-1] + h1[A[i-1]]
        prefix2_A[i] = prefix2_A[i-1] + h2[A[i-1]]

    prefix1_B = [0] * (N + 1)
    prefix2_B = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix1_B[i] = prefix1_B[i-1] + h1[B[i-1]]
        prefix2_B[i] = prefix2_B[i-1] + h2[B[i-1]]

    # Process each query
    output = []
    for _ in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        lenA = r - l + 1
        lenB = R - L + 1
        if lenA != lenB:
            output.append("No")
            continue

        a1 = prefix1_A[r] - prefix1_A[l-1]
        a2 = prefix2_A[r] - prefix2_A[l-1]
        b1 = prefix1_B[R] - prefix1_B[L-1]
        b2 = prefix2_B[R] - prefix2_B[L-1]

        if a1 == b1 and a2 == b2:
            output.append("Yes")
        else:
            output.append("No")

    print('
'.join(output))

if __name__ == '__main__':
    main()