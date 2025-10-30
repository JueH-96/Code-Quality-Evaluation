def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # Read grid A
    A = data[1 : 1 + N]
    # Read grid B
    B = data[1 + N : 1 + 2*N]

    for i in range(N):
        for j in range(N):
            if A[i][j] != B[i][j]:
                print(i+1, j+1)  # +1 for 1-based indexing
                return

# Call the main function
if __name__ == "__main__":
    main()