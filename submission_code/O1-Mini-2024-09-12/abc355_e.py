import sys

def main():
    import sys

    def get_max_k(L, R, N):
        if L == 0:
            k = N
        else:
            k = (L & -L).bit_length() - 1
        while k > 0 and (L + (1 << k) - 1) > R:
            k -= 1
        return k

    input_line = sys.stdin.readline()
    while input_line.strip() == '':
        input_line = sys.stdin.readline()
    N_L_R = input_line.strip().split()
    if len(N_L_R) < 3:
        N_L_R += sys.stdin.readline().strip().split()
    N, L, R = map(int, N_L_R)
    S = 0
    while L <= R:
        k = get_max_k(L, R, N)
        j = L // (1 << k)
        print(f"? {k} {j}", flush=True)
        try:
            T_line = sys.stdin.readline()
            if not T_line:
                sys.exit()
            T = int(T_line.strip())
            if T == -1:
                sys.exit()
            S = (S + T) % 100
            L += (1 << k)
        except:
            sys.exit()
    print(f"! {S}", flush=True)

if __name__ == "__main__":
    main()