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

    # Generate hash values for each possible element
    max_val = N
    H1 = [0] * (max_val + 2)
    H2 = [0] * (max_val + 2)
    random.seed(1983)  # Fixed seed for reproducibility
    for v in range(1, max_val + 1):
        H1[v] = random.getrandbits(64)
        H2[v] = random.getrandbits(64)

    # Precompute prefix sums for A and B using both hash functions
    prefixA1 = [0] * (N + 1)
    prefixA2 = [0] * (N + 1)
    for i in range(1, N + 1):
        a = A[i - 1]
        prefixA1[i] = prefixA1[i - 1] + H1[a]
        prefixA2[i] = prefixA2[i - 1] + H2[a]

    prefixB1 = [0] * (N + 1)
    prefixB2 = [0] * (N + 1)
    for i in range(1, N + 1):
        b = B[i - 1]
        prefixB1[i] = prefixB1[i - 1] + H1[b]
        prefixB2[i] = prefixB2[i - 1] + H2[b]

    res = []
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr + 1])
        L = int(input[ptr + 2])
        R = int(input[ptr + 3])
        ptr += 4

        len_a = r - l + 1
        len_b = R - L + 1
        if len_a != len_b:
            res.append("No")
            continue

        sumA1 = prefixA1[r] - prefixA1[l - 1]
        sumA2 = prefixA2[r] - prefixA2[l - 1]
        sumB1 = prefixB1[R] - prefixB1[L - 1]
        sumB2 = prefixB2[R] - prefixB2[L - 1]

        if sumA1 == sumB1 and sumA2 == sumB2:
            res.append("Yes")
        else:
            res.append("No")

    print('
'.join(res))

if __name__ == "__main__":
    main()