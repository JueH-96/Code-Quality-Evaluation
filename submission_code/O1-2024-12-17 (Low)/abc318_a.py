def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M, P = map(int, data)

    if M > N:
        print(0)
        return

    # Calculate how many full moons occur from day M to N with a step of P
    # The count of terms in an arithmetic progression from M to at most N with difference P
    # is given by 1 + (N - M) // P, provided M <= N.
    count = 1 + (N - M) // P
    print(count)

# Do not forget to call the main function
if __name__ == "__main__":
    main()