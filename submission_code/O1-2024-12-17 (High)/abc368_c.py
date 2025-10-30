def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    H = list(map(int, data[1:]))

    offset = 0  # This will track the total number of attacks (T)
    for h in H:
        # offset_mod is offset % 3, to see how the 3-damage pattern aligns
        offset_mod = offset % 3
        # We define L = needed health + offset_mod; see explanation
        L = h + offset_mod

        best_x = None

        # We try all possible remainders r in [0,1,2] for x = 3*q + r
        for r in (0, 1, 2):
            # Condition 1: 5*q + r >= L  => q >= (L - r + 4)//5 if L>r, else 0
            if L <= r:
                q1 = 0
            else:
                q1 = (L - r + 4) // 5

            # Condition 2: 3*q + r >= offset_mod + 1 => q >= (offset_mod+1 - r + 2)//3 if offset_mod+1>r, else 0
            if offset_mod + 1 <= r:
                q2 = 0
            else:
                need = offset_mod + 1 - r
                if need <= 0:
                    q2 = 0
                else:
                    q2 = (need + 2) // 3

            q = max(q1, q2)
            x = 3 * q + r  # x is the T-value offset portion

            if best_x is None or x < best_x:
                best_x = x

        # k is how many attacks this enemy needs from the current offset
        k = best_x - offset_mod
        offset += k

    print(offset)

# Do not remove or rename this call
if __name__ == "__main__":
    main()