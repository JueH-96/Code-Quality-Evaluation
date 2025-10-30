def main() -> None:
    import sys

    # Read input
    W, B = map(int, sys.stdin.readline().split())
    L = W + B                       # length of the required substring
    pattern = "wbwbwwbwbwbw"        # one period of the infinite keyboard
    P = len(pattern)                # = 12

    # Repeat the pattern enough times so that every substring of length L
    # starting at any offset (0..P-1) is fully contained in the finite string.
    # It suffices to make the length at least L + P.
    repeats = (L // P) + 3          # definitely long enough (>= L + P)
    S = pattern * repeats
    n = len(S)

    # Prefix sum of the number of white keys up to each position
    pref_w = [0] * (n + 1)
    for i, ch in enumerate(S):
        pref_w[i + 1] = pref_w[i] + (1 if ch == 'w' else 0)

    # Scan all substrings of length L
    for start in range(n - L + 1):
        whites = pref_w[start + L] - pref_w[start]
        if whites == W:
            # then blacks = L - whites == B automatically
            print("Yes")
            return

    print("No")


if __name__ == "__main__":
    main()