def main():
    import sys
    data = sys.stdin.read().strip().split()
    S, T = data[0], data[1]

    positions = []
    i, j = 0, 0  # Pointers for S and T respectively
    while i < len(S) and j < len(T):
        if S[i] == T[j]:
            positions.append(j+1)  # Store 1-based index
            i += 1
        j += 1

    print(" ".join(map(str, positions)))

# Do not remove the function call below
main()