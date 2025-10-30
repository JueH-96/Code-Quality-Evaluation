MOD = 998244353

def modinv(a):
    return pow(a, MOD - 2, MOD)

def fenwick_sum(fen, i):
    res = 0
    while i > 0:
        res += fen[i]
        i -= i & -i
    return res

def fenwick_add(fen, i, x, n):
    while i <= n:
        fen[i] = (fen[i] + x) % MOD
        i += i & -i

def compute_inversion(a, n):
    fen = [0] * (n + 2)
    inv = 0
    for i in range(n):
        inv = (inv + fenwick_sum(fen, a[i] - 1)) % MOD
        fenwick_add(fen, a[i], 1, n)
    return inv

def compute_sum_inversion_window(a, n, K):
    fen = [0] * (K + 2)
    sum_inv = 0
    for i in range(K):
        sum_inv = (sum_inv + fenwick_sum(fen, a[i] - 1)) % MOD
        fenwick_add(fen, a[i], 1, K)
    for i in range(K, n):
        # Remove a[i-K]
        fenwick_add(fen, a[i - K], -1, K)
        # Add a[i]
        sum_inv = (sum_inv + fenwick_sum(fen, a[i] - 1)) % MOD
        fenwick_add(fen, a[i], 1, K)
    return sum_inv

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    M = N - K + 1
    if M <= 0:
        print(0)
        return
    
    # Compute original inversion count
    Inv_P = compute_inversion(P, N)
    
    # Compute sum of inversion counts over all windows of size K
    sum_Inv_S = compute_sum_inversion_window(P, N, K)
    
    # Compute E[Inv_S] = sum_Inv_S / M
    E_Inv_S = (sum_Inv_S * modinv(M)) % MOD
    
    # Compute expected inversions within S after shuffling: K*(K-1)/4
    expected_inversions_shuffled = (K * (K - 1) // 4) % MOD
    
    # Compute E[Inversion(P')] = Inv(P) - E[Inv_S] + expected_inversions_shuffled
    E_Inversion_P_prime = (Inv_P - E_Inv_S + expected_inversions_shuffled) % MOD
    
    # Compute the modular inverse of 1 (since no division needed here)
    # But ensure it's within MOD
    E_Inversion_P_prime = E_Inversion_P_prime % MOD
    
    print(E_Inversion_P_prime)

if __name__ == "__main__":
    main()