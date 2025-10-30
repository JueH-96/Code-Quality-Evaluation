MOD = 998244353

def modpow(x, n):
    if n == 0:
        return 1
    elif n % 2 == 1:
        return x * modpow(x, n - 1) % MOD
    else:
        y = modpow(x, n // 2)
        return y * y % MOD

def modinv(x):
    return modpow(x, MOD - 2)

N, X = map(int, input().split())
T = list(map(int, input().split()))

total_time = sum(T)
prob_song1 = T[0] * modinv(total_time)

if X < T[0]:
    print(modpow(2, N) * prob_song1 % MOD)
else:
    num_cycles = X // total_time
    remaining_time = X % total_time

    if remaining_time < T[0]:
        result = (modpow(2, N) * modpow(N, num_cycles) * prob_song1) % MOD
    else:
        result = (modpow(2, N) * modpow(N, num_cycles) * modpow(N - 1, 1) * modinv(N)) % MOD

    print(result)