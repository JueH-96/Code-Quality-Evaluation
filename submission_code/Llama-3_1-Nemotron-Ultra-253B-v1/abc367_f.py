import sys
import random

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+N]))
    ptr += N
    B = list(map(int, data[ptr:ptr+N]))
    ptr += N

    # Generate random hash values for each possible element (1-based)
    max_x = N
    random.seed(42)
    hash_val = [0] * (max_x + 1)
    for x in range(1, max_x + 1):
        hash_val[x] = random.getrandbits(64)

    # Compute prefix sums for A and B
    prefix_A = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_A[i] = prefix_A[i-1] + hash_val[A[i-1]]

    prefix_B = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_B[i] = prefix_B[i-1] + hash_val[B[i-1]]

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

        len_A = r - l + 1
        len_B = R - L + 1
        if len_A != len_B:
            output.append("No")
        else:
            sum_A = prefix_A[r] - prefix_A[l-1]
            sum_B = prefix_B[R] - prefix_B[L-1]
            if sum_A == sum_B:
                output.append("Yes")
            else:
                output.append("No")
    
    print('
'.join(output))

if __name__ == "__main__":
    main()