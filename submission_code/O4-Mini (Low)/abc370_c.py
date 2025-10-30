def main():
    import sys

    S = list(sys.stdin.readline().strip())
    T = list(sys.stdin.readline().strip())
    N = len(S)

    result = []
    # Repeat until S matches T
    while S != T:
        # Find all positions where S and T differ
        diffs = [i for i in range(N) if S[i] != T[i]]
        # Among these, find positions where moving makes the string lexicographically smaller:
        # i.e. T[i] < S[i]
        downs = [i for i in diffs if T[i] < S[i]]
        if downs:
            # pick the smallest such index
            idx = downs[0]
        else:
            # all moves would increase lex; defer as far right as possible
            idx = diffs[-1]
        # Perform the change
        S[idx] = T[idx]
        result.append(''.join(S))

    # Output
    print(len(result))
    for line in result:
        print(line)


if __name__ == "__main__":
    main()