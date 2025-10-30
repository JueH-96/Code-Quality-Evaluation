def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # The total sum over all N players' scores must remain zero.
    # Given A[0..N-2] for players 1..N-1, the N-th player's score is
    # the negation of their sum.
    result = -sum(A)
    print(result)

if __name__ == "__main__":
    main()