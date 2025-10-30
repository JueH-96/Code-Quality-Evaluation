# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def modinv(a, p):
    return pow(a, p - 2, p)

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Calculate the expected value
    expected_value = 0
    for i in range(1, N + 1):
        expected_value += A[i - 1] * modinv(i, MOD)
        expected_value %= MOD
    
    # Multiply by N to get the final expected value
    expected_value = (expected_value * N) % MOD
    
    print(expected_value)

if __name__ == "__main__":
    main()