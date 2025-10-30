import sys

MODULUS = 998244353

def main():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    trailing_zeros, digit_sum = [0], [0]
    for a in A[::-1]:
        t = len(str(a)) - 1
        trailing_zeros.append((trailing_zeros[-1] + pow(10, t, MODULUS)) % MODULUS)
        digit_sum.append((digit_sum[-1] + a * pow(10, t, MODULUS)) % MODULUS)

    answer = 0
    for i, a in enumerate(A):
        answer += (trailing_zeros[N-i] * a + digit_sum[N-i] * pow(10, len(str(a)), MODULUS)) % MODULUS

    print(answer * 2 % MODULUS)

if __name__ == "__main__":
    main()