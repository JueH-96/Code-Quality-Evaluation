def main():
    import sys
    data = sys.stdin.read().strip().split()
    W, B = map(int, data)
    L = W + B
    # The base repeating pattern
    pattern = "wbwbwwbwbwbw"
    # We need to cover any substring of length up to L in the infinite repetition.
    # Since the pattern length is 12, generating ~30 repeats (360 chars) is enough
    # for L up to 200 (max W+B = 200).
    S = pattern * 30
    n = len(S)
    # Build prefix sums of w's
    pre_w = [0] * (n + 1)
    for i, ch in enumerate(S):
        pre_w[i+1] = pre_w[i] + (1 if ch == 'w' else 0)
    # Slide a window of length L
    for i in range(n - L + 1):
        cnt_w = pre_w[i+L] - pre_w[i]
        cnt_b = L - cnt_w
        if cnt_w == W and cnt_b == B:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()