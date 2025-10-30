import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Compute prefix sums S and mod_S
    S = [0] * (N + 1)
    for i in range(1, N + 1):
        S[i] = S[i-1] + A[i-1]
    mod_S = [0] * (N + 1)
    for i in range(N + 1):
        mod_S[i] = S[i] % M

    T_mod = S[N] % M

    # Calculate case1
    case1 = 0
    freq_case1 = [0] * M
    for i in range(N):  # 0 to N-1 inclusive
        x = mod_S[i]
        case1 += freq_case1[x]
        freq_case1[x] += 1

    # Calculate case2
    total_case2 = 0
    freq_case2 = [0] * M
    for i in range(N-1, -1, -1):
        current_mod = mod_S[i]
        target_mod = (current_mod + T_mod) % M
        total_case2 += freq_case2[target_mod]
        freq_case2[current_mod] += 1

    print(case1 + total_case2)

if __name__ == "__main__":
    main()