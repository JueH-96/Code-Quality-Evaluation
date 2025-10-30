MOD = 998244353

def main():
    N = int(input())
    A = list(map(int, input().split()))
    inv_N = pow(N, MOD-2, MOD)
    prev_S = 0
    prev_E = 0
    for x in range(N-1, -1, -1):
        current_A = A[x]
        term = (current_A + prev_E) % MOD
        current_S = (term + prev_S) % MOD
        E_x = (current_S * inv_N) % MOD
        prev_S, prev_E = current_S, E_x
    print(prev_E)

if __name__ == "__main__":
    main()