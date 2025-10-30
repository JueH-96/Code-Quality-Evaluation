MOD = 998244353

def f(x, y):
    return int(str(x) + str(y))

def main():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Precompute the powers of 10 needed to calculate f(x, y) efficiently
    max_len = max(len(str(a)) for a in A)
    power_of_10 = [1] * (max_len + 1)
    for i in range(1, max_len + 1):
        power_of_10[i] = (power_of_10[i-1] * 10) % MOD
    
    # Calculate the contribution of each A[i] to the final sum
    contribution = [0] * N
    for i in range(N):
        len_Ai = len(str(A[i]))
        contribution[i] = (A[i] * power_of_10[len_Ai]) % MOD
    
    # Calculate the sum of all f(A[i], A[j]) for i < j
    result = 0
    for i in range(N):
        for j in range(i+1, N):
            result += (contribution[i] + A[j]) % MOD
            result %= MOD
    
    print(result)

if __name__ == "__main__":
    main()