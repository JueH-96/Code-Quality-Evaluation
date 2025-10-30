import sys

def can_be_non_decreasing(N, A):
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + A[i - 1]
    
    B_prev = 0
    for i in range(1, N + 1):
        current = max(prefix[i], B_prev)
        if current < prefix[i]:
            return False
        B_prev = current
    return True

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N = int(input[idx])
        idx += 1
        A = list(map(int, input[idx:idx + N]))
        idx += N
        if can_be_non_decreasing(N, A):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()