def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    # Compute prefix sums
    S = [0] * (N + 1)
    for i in range(N):
        S[i + 1] = S[i] + A[i]

    total_perimeter = S[N]
    mod_perimeter = total_perimeter % M

    # Build array C where C[i] = S[i] mod M, for i in [0 .. N-1]
    # (S[i] is the sum of the first i elements of A)
    C = [S[i] % M for i in range(N)]

    # Case 1: t >= s  <=>  (i < j) and C[i] == C[j]
    freq = [0] * M  # frequency of each remainder
    count_case1 = 0
    for i in range(N - 1, -1, -1):
        count_case1 += freq[C[i]]
        freq[C[i]] += 1

    # Case 2: t < s  <=>  (j < i) and C[j] == (C[i] - mod_perimeter) mod M
    freq2 = [0] * M
    count_case2 = 0
    for i in range(N):
        need = (C[i] - mod_perimeter) % M
        count_case2 += freq2[need]
        freq2[C[i]] += 1

    print(count_case1 + count_case2)

# Call main() at the end
if __name__ == "__main__":
    main()