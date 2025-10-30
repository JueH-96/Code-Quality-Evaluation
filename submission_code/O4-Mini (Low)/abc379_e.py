def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]
    # f is the sum of all substrings ending at the current position i
    # ans accumulates the total over all ending positions.
    f = 0
    ans = 0
    for i, ch in enumerate(S, start=1):
        d = int(ch)
        # Extend all previous substrings by this digit (multiply by 10)
        # and add the new substrings that consist of this digit appended
        # to each possible start among the first i positions => d*i.
        f = f * 10 + d * i
        ans += f
    print(ans)

if __name__ == "__main__":
    main()