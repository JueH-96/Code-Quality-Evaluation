import sys
import numpy as np

def main():
    data = sys.stdin.read().split()
    idx = 0

    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N

    M = int(data[idx])
    idx += 1
    B = list(map(int, data[idx:idx+M]))
    idx += M

    L = int(data[idx])
    idx += 1
    C = list(map(int, data[idx:idx+L]))
    idx += L

    Q = int(data[idx])
    idx += 1
    X = list(map(int, data[idx:idx+Q]))

    # Convert to sets to eliminate duplicates
    A = list(set(A))
    B = list(set(B))
    C = list(set(C))

    # Convert to NumPy arrays
    A_np = np.array(A, dtype=np.int64)
    B_np = np.array(B, dtype=np.int64)
    C_np = np.array(C, dtype=np.int64)

    # Compute sum of A and B
    sum_ab = A_np[:, None] + B_np[None, :]
    sum_ab_flat = sum_ab.flatten()

    # Create a set of A+B sums
    sum_ab_set = set(sum_ab_flat)

    # For each query X_i, check if X_i - c is in sum_ab_set for any c in C
    answers = []
    for x in X:
        found = False
        for c in C:
            if (x - c) in sum_ab_set:
                found = True
                break
        answers.append("Yes" if found else "No")

    # Print all answers
    print('
'.join(answers))

if __name__ == '__main__':
    main()