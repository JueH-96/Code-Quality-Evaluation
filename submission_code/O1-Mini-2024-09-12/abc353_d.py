import sys

def main():
    MOD = 998244353

    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))

    # Precompute number of digits
    digits = [0]*N
    for idx in range(N):
        x = A[idx]
        if x < 10:
            digits[idx] =1
        elif x < 100:
            digits[idx] =2
        elif x < 1000:
            digits[idx] =3
        elif x < 10000:
            digits[idx] =4
        elif x < 100000:
            digits[idx] =5
        elif x < 1000000:
            digits[idx] =6
        elif x < 10000000:
            digits[idx] =7
        elif x < 100000000:
            digits[idx] =8
        elif x < 1000000000:
            digits[idx] =9
        else:
            digits[idx] =10

    # Precompute powers of 10
    pow10 = [1]*11
    for k in range(1,11):
        pow10[k] = (pow10[k-1] *10) % MOD

    # Precompute suffix_sum_10k
    suffix_sum = [0]*(N+1)
    for i in range(N-1, -1, -1):
        suffix_sum[i] = (suffix_sum[i+1] + pow10[digits[i]]) % MOD

    # Compute first sum
    first_sum =0
    for i in range(N-1):
        first_sum = (first_sum + A[i] * suffix_sum[i+1]) % MOD

    # Compute second sum
    second_sum =0
    for j in range(1, N):
        second_sum = (second_sum + A[j] * j) % MOD

    total = (first_sum + second_sum) % MOD
    print(total)

if __name__ == "__main__":
    main()