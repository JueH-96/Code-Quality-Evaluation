import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    A = [0] + A  # 1-based indexing

    delta = [0] * (N + 2)  # To handle up to N+1
    current_contrib = 0
    R = [0] * (N + 1)
    S = [0] * (N + 1)
    G = [0] * (N + 1)

    for j in range(1, N + 1):
        current_contrib += delta[j]
        R[j] = current_contrib
        S[j] = A[j] + R[j]
        G[j] = min(S[j], N - j)

        a = j + 1
        upper_end1 = j + G[j] + 1
        upper_end2 = j + S[j]
        b = min(upper_end1, upper_end2, N)

        if a <= b:
            delta[a] += 1
            delta[b + 1] -= 1

    B = [0] * (N + 1)
    for j in range(1, N + 1):
        B[j] = S[j] - G[j]

    print(' '.join(map(str, B[1:N+1])))

if __name__ == "__main__":
    main()