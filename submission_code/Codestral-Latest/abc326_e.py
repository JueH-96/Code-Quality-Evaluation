import sys
from fractions import Fraction

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    # Calculate the expected value of Aoki's salary
    expected_value = Fraction(0)
    for i in range(1, N + 1):
        expected_value += Fraction(A[i - 1], N) / i

    # Convert the expected value to an integer modulo 998244353
    result = (expected_value.numerator * pow(expected_value.denominator, -1, MOD)) % MOD

    print(result)

if __name__ == "__main__":
    main()