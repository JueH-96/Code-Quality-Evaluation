def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N-1]))
    ptr += N-1

    A.sort()
    B.sort()

    # Compute k_a
    k_a = 0
    for i in range(len(B)):
        if B[i] >= A[i]:
            k_a += 1
        else:
            break

    # Compute k_b
    max_j = -1
    for j in range(len(B)):
        if B[j] < A[j+1]:
            max_j = j
    k_b = max_j + 1

    if k_b > k_a:
        print(-1)
        return

    min_x = float('inf')
    for k in range(k_b, k_a + 1):
        if k == 0:
            x_candidate = A[0]
            if len(B) >= 1 and x_candidate <= B[0]:
                min_x = min(min_x, x_candidate)
        elif k < len(B):
            x_candidate = max(A[k], B[k-1])
            if x_candidate <= B[k]:
                min_x = min(min_x, x_candidate)
        else:
            x_candidate = max(A[k], B[-1])
            min_x = min(min_x, x_candidate)

    if min_x == float('inf'):
        print(-1)
    else:
        print(min_x)

if __name__ == '__main__':
    main()