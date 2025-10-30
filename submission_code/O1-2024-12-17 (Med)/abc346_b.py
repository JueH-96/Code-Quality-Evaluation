def main():
    import sys
    data = sys.stdin.read().strip().split()
    W, B = map(int, data)

    # The repeating pattern:
    pattern = "wbwbwwbwbwbw"
    # Repeat the pattern sufficiently many times to cover any substring up to length W+B
    # Since W+B <= 200, let's just make a string of length >= 200 + 12 = 212.
    # 25 repeats => length 300, which is comfortably large.
    repeats = 25
    S = pattern * repeats

    total_length = W + B
    # Edge case: if total_length == 0 (though constraints say W+B >= 1), would be trivial.
    # We'll assume W+B >= 1 as per constraints.

    # Build prefix sums for 'w' and 'b'
    prefix_w = [0] * (len(S)+1)
    prefix_b = [0] * (len(S)+1)

    for i, ch in enumerate(S):
        prefix_w[i+1] = prefix_w[i] + (1 if ch == 'w' else 0)
        prefix_b[i+1] = prefix_b[i] + (1 if ch == 'b' else 0)

    # Check all substrings of length W+B
    found = False
    for start in range(len(S) - total_length + 1):
        w_count = prefix_w[start + total_length] - prefix_w[start]
        b_count = prefix_b[start + total_length] - prefix_b[start]
        if w_count == W and b_count == B:
            found = True
            break

    print("Yes" if found else "No")


# Do not forget to call main()
if __name__ == "__main__":
    main()