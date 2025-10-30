def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S = data[0]
    T = data[1]

    len_S = len(S)
    len_T = len(T)
    i = 0  # Pointer for S
    j = 0  # Pointer for T
    positions = []

    while i < len_S and j < len_T:
        if S[i] == T[j]:
            positions.append(j + 1)  # 1-based indexing
            i += 1
        j += 1

    print(' '.join(map(str, positions)))

if __name__ == "__main__":
    main()