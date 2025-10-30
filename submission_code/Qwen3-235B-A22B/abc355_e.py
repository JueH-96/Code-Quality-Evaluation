import sys

def main():
    import sys

    N, L, R = map(int, sys.stdin.readline().split())
    blocks = []
    current_L = L
    current_R = R

    while current_L <= current_R:
        if current_L == 0:
            i_max = N
        else:
            t = current_L & (-current_L)
            trailing_power = 0
            while (t >> trailing_power) & 1 == 0:
                trailing_power += 1
            i_max = trailing_power

        length = current_R - current_L + 1
        remaining_max_power = (length).bit_length() - 1 if length > 0 else 0
        k = min(i_max, remaining_max_power)
        block_size = 1 << k
        j = current_L // block_size
        blocks.append((k, j))
        current_L += block_size

    sum_mod = 0
    for i, j in blocks:
        print(f"? {i} {j}")
        sys.stdout.flush()
        T = int(sys.stdin.readline())
        if T == -1:
            return
        sum_mod = (sum_mod + T) % 100

    print(f"! {sum_mod}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()