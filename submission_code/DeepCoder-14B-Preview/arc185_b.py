import sys

def can_make_non_decreasing(N, A):
    prefix = [0] * N
    prefix[0] = A[0]
    for i in range(1, N):
        prefix[i] = prefix[i-1] + A[i]
    total = prefix[-1] if N > 0 else 0

    if N == 0:
        return True

    B = [0] * N
    B[0] = prefix[0]
    current_sum = B[0]

    for i in range(1, N):
        min_b = max(B[i-1], prefix[i] - current_sum)
        B[i] = min_b
        current_sum += min_b

    if current_sum > total:
        return False

    rem = total - current_sum
    B[-1] += rem

    for i in range(1, N):
        if B[i] < B[i-1]:
            return False

    return True

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        if can_make_non_decreasing(N, A):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()