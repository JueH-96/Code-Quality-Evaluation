import sys

MOD = 998244353

def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    N = int(input_data[0])
    A = [0] * (N + 1)                     # 1-based
    for i in range(N):
        A[i + 1] = int(input_data[i + 1])

    invN = pow(N, MOD - 2, MOD)           # modular inverse of N (MOD is prime)

    E = [0] * (N + 1)                     # E[N] is already 0
    suffix_sum = 0                        # Σ_{y = x+1..N} (A_y + E[y])

    for x in range(N - 1, -1, -1):
        y = x + 1                         # the element that enters the suffix
        suffix_sum = (suffix_sum + A[y] + E[y]) % MOD
        E[x] = suffix_sum * invN % MOD

    print(E[0] % MOD)


if __name__ == "__main__":
    main()