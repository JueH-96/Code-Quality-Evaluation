import sys

def main() -> None:
    # Fast input
    input_data = sys.stdin.buffer.read().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    max_val = max(A)

    # Frequency table
    freq = [0] * (max_val + 1)
    for v in A:
        freq[v] += 1

    # Prefix sums of frequencies
    prefix = [0] * (max_val + 1)
    s = 0
    for i in range(1, max_val + 1):
        s += freq[i]
        prefix[i] = s

    ans = 0

    # Iterate over the possible smaller value a
    for a in range(1, max_val + 1):
        c = freq[a]
        if c == 0:
            continue

        # Pairs where the two elements are equal (ratio = 1)
        ans += c * (c - 1) // 2  # floor(a / a) == 1

        # Pairs with the larger element strictly larger than a
        # floor(b / a) == k for b in [k*a, (k+1)*a - 1]
        max_k = max_val // a
        for k in range(1, max_k + 1):
            low = k * a
            high = (k + 1) * a - 1
            if high > max_val:
                high = max_val

            # We need b > a as well
            start = max(a + 1, low)
            if start > high:
                continue

            cnt_b = prefix[high] - prefix[start - 1]
            if cnt_b:
                ans += c * cnt_b * k

    print(ans)


if __name__ == "__main__":
    main()