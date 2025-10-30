def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1

    H = input[idx]
    idx += 1
    instructions = []
    for _ in range(Q):
        T = input[idx]
        idx += 1
        instructions.append((H, T))
        H = input[idx]
        idx += 1

    L_pos = 1
    R_pos = 2
    total_steps = 0

    for H_i, T_i in instructions:
        a = L_pos if H_i == 'L' else R_pos
        b = int(T_i)
        c = R_pos if H_i == 'L' else L_pos

        # Compute d1: minimal steps between a and b
        d1 = min(abs(b - a), N - abs(b - a))

        # Compute d_ac: minimal steps between a and c
        d_ac = min(abs(c - a), N - abs(c - a))

        # Compute d_cb: minimal steps between c and b
        d_cb = min(abs(b - c), N - abs(b - c))

        if (d_ac + d_cb) == d1:
            total_steps += (N - d1)
        else:
            total_steps += d1

        # Update the position of the required hand
        if H_i == 'L':
            L_pos = b
        else:
            R_pos = b

    print(total_steps)

if __name__ == "__main__":
    main()