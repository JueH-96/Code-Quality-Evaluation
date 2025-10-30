def main():
    S = input().strip()
    N = len(S)
    i = 0  # index in S
    j = 0  # index in the infinite pattern IOIO...
    # advance j until we match all chars of S
    while i < N:
        expected = 'i' if (j % 2) == 0 else 'o'
        if S[i] == expected:
            i += 1
        j += 1
    # Now j is the length of the pattern we used. It must be even.
    if j % 2 == 1:
        j += 1
    # number of insertions = total pattern length - original length
    print(j - N)

if __name__ == "__main__":
    main()