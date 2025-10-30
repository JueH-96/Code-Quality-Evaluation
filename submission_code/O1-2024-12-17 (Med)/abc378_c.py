def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    last_occurrence = {}
    B = []

    for i in range(N):
        if A[i] in last_occurrence:
            B.append(last_occurrence[A[i]])
        else:
            B.append(-1)
        last_occurrence[A[i]] = i + 1  # Store 1-based index

    print(" ".join(map(str, B)))

# Do not remove the function call below
main()