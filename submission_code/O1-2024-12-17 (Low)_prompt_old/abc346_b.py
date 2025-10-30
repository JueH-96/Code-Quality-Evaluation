def solve():
    import sys

    data = sys.stdin.read().strip().split()
    W, B = map(int, data)

    # The repeating pattern (length = 12):
    pattern = "wbwbwwbwbwbw"

    # To handle substrings that might span across boundaries,
    # we'll repeat the pattern enough times so that any substring
    # of length up to (W+B) is fully contained within this extended string.
    # W+B can be up to 200, so repeating 12*20=240 characters is enough.
    repeat_count = 20
    bigS = pattern * repeat_count

    # Prefix sums for counting 'w' and 'b'
    n = len(bigS)
    prefixW = [0] * (n + 1)
    prefixB = [0] * (n + 1)

    for i in range(n):
        prefixW[i+1] = prefixW[i] + (1 if bigS[i] == 'w' else 0)
        prefixB[i+1] = prefixB[i] + (1 if bigS[i] == 'b' else 0)

    # Check all substrings
    for start in range(n):
        for end in range(start + 1, n + 1):
            w_count = prefixW[end] - prefixW[start]
            b_count = prefixB[end] - prefixB[start]
            if w_count == W and b_count == B:
                print("Yes")
                return

    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()