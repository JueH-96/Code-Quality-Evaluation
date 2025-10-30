import sys

data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1

for test in range(T):
    N = int(data[index])
    X = int(data[index + 1])
    K = int(data[index + 2])
    index += 3

    Dx = X.bit_length() - 1
    D_max = N.bit_length() - 1
    ans = 0

    for L in range(0, Dx + 1):
        Dv = K - Dx + 2 * L
        if Dv < 0 or Dv > D_max:
            continue
        if Dv < L:
            continue
        if Dv == L:
            if K == Dx - L:
                ans += 1
        else:  # Dv > L
            if L < Dx:
                A_L_val = X >> (Dx - L)
                A_next_val = X >> (Dx - L - 1)
                if A_next_val == (2 * A_L_val):
                    B_L_val = 2 * A_L_val + 1
                else:
                    B_L_val = 2 * A_L_val
                if 1 <= B_L_val <= N:
                    D_rel = Dv - (L + 1)
                    left = (B_L_val << D_rel)
                    right = left + (1 << D_rel) - 1
                    num = max(0, min(right, N) - left + 1)
                    ans += num
            else:  # L == Dx and Dv > L
                D_rel = Dv - Dx  # equals K
                Q_val = X
                left = (Q_val << D_rel)
                right = left + (1 << D_rel) - 1
                num = max(0, min(right, N) - left + 1)
                ans += num

    print(ans)