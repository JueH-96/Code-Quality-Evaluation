modulus = 998244353
import math
import sys

def fwht(arr, U):
    n = 20  # Because U = 2^20
    for i in range(n):
        step = 1 << (i + 1)
        bit = 1 << i
        for j in range(0, U, step):
            for k in range(j, j + bit):
                x = arr[k]
                y = arr[k + bit]
                arr[k] = (x + y) % modulus
                arr[k + bit] = (x - y) % modulus

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3 + N]))
    U = 1 << 20  # 2^20

    # Step 1: Compute T = FWHT(G), where G[x] = x^K
    G_arr = [0] * U
    for x in range(U):
        G_arr[x] = pow(x, K, modulus)
    fwht(G_arr, U)
    T_arr = G_arr

    # Step 2: Compute C and then Chat = FWHT(C)
    C_arr = [0] * U
    for a in A:
        C_arr[a] = (C_arr[a] + 1) % modulus
    Chat_arr = C_arr[:]
    fwht(Chat_arr, U)

    # Step 3: Precompute roots and power tables
    d = math.gcd(M, modulus - 1)
    g = 3
    w0 = pow(g, (modulus - 1) // d, modulus)
    roots = []
    w = 1
    for j in range(d):
        roots.append(w)
        w = (w * w0) % modulus

    A_powers = []
    B_powers = []
    for j in range(d):
        a_val = (1 + roots[j]) % modulus
        b_val = (1 - roots[j]) % modulus
        tab_a = [1] * (N + 1)
        for i in range(1, N + 1):
            tab_a[i] = tab_a[i - 1] * a_val % modulus
        tab_b = [1] * (N + 1)
        for i in range(1, N + 1):
            tab_b[i] = tab_b[i - 1] * b_val % modulus
        A_powers.append(tab_a)
        B_powers.append(tab_b)
    inv_d = pow(d, -1, modulus)

    # Step 4: Compute F0[s] for each s
    F0_arr = [0] * U
    for s in range(U):
        c_val = Chat_arr[s]
        if c_val > (modulus - 1) // 2:
            c_val -= modulus
        total = N + c_val
        p_int = total // 2
        if p_int < 0:
            p_int = 0
        elif p_int > N:
            p_int = N
        q_int = N - p_int
        res = 0
        for j in range(d):
            a_val = A_powers[j][p_int]
            b_val = B_powers[j][q_int]
            res = (res + a_val * b_val) % modulus
        F0_arr[s] = res * inv_d % modulus

    # Step 5: Compute the answer
    total_sum = 0
    for s in range(U):
        total_sum = (total_sum + F0_arr[s] * T_arr[s]) % modulus
    invU = pow(U, -1, modulus)
    total_sum = total_sum * invU % modulus
    print(total_sum)

if __name__ == '__main__':
    main()