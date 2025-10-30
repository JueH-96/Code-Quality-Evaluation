import sys
import random

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1

    A = list(map(int, data[ptr:ptr+N]))
    ptr += N
    B = list(map(int, data[ptr:ptr+N]))
    ptr += N

    # Generate two sets of random 64-bit integers for hashing
    max_val = N
    hash1 = [0] * (max_val + 1)
    hash2 = [0] * (max_val + 1)
    random.seed(42)  # Fixed seed for reproducibility
    for v in range(1, max_val + 1):
        hash1[v] = random.getrandbits(64)
        hash2[v] = random.getrandbits(64)

    # Compute prefix sums for both hash functions in A and B
    prefix_A1 = [0] * (N + 1)
    prefix_A2 = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_A1[i] = prefix_A1[i-1] + hash1[A[i-1]]
        prefix_A2[i] = prefix_A2[i-1] + hash2[A[i-1]]

    prefix_B1 = [0] * (N + 1)
    prefix_B2 = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_B1[i] = prefix_B1[i-1] + hash1[B[i-1]]
        prefix_B2[i] = prefix_B2[i-1] + hash2[B[i-1]]

    # Process each query
    output = []
    for _ in range(Q):
        l = int(data[ptr])
        ptr += 1
        r = int(data[ptr])
        ptr += 1
        L = int(data[ptr])
        ptr += 1
        R = int(data[ptr])
        ptr += 1

        sum_a1 = prefix_A1[r] - prefix_A1[l-1]
        sum_b1 = prefix_B1[R] - prefix_B1[L-1]
        sum_a2 = prefix_A2[r] - prefix_A2[l-1]
        sum_b2 = prefix_B2[R] - prefix_B2[L-1]

        if sum_a1 == sum_b1 and sum_a2 == sum_b2:
            output.append("Yes")
        else:
            output.append("No")

    print('
'.join(output))

if __name__ == '__main__':
    main()