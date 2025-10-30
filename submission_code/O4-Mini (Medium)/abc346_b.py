def main():
    import sys

    data = sys.stdin.read().split()
    W = int(data[0])
    B = int(data[1])
    L = W + B

    # The base pattern of length 12
    pattern = "wbwbwwbwbwbw"
    P = len(pattern)
    tot_w = pattern.count('w')

    # Build prefix sums over two copies of the pattern
    # pre[i] = number of 'w' in pattern[0:i]
    pre = [0] * (2 * P + 1)
    for i in range(2 * P):
        pre[i + 1] = pre[i] + (1 if pattern[i % P] == 'w' else 0)

    # Number of full repeats in a length-L substring
    full = L // P
    rem = L % P

    # Try every possible alignment shift 0..P-1
    for start in range(P):
        # w's from full repeats
        cnt_w = full * tot_w
        # w's from the remaining rem characters
        # they run from index 'start' to 'start+rem-1' (mod P)
        cnt_w += pre[start + rem] - pre[start]
        if cnt_w == W:
            print("Yes")
            return

    print("No")


if __name__ == "__main__":
    main()