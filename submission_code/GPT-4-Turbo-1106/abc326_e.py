MOD = 998244353

def modinv(a, mod=MOD):
    return pow(a, mod - 2, mod)

def main():
    N = int(input())
    A = list(map(int, input().split()))

    # Calculate the expected value
    expected_value = 0
    probability = modinv(N)
    for i in range(N - 1, -1, -1):
        expected_value = (expected_value + A[i] * probability) % MOD
        probability = probability * (i + 1) * modinv(i + 2) % MOD

    print(expected_value)

if __name__ == "__main__":
    main()