def main():
    import sys

    # Read input
    N, T, A = map(int, sys.stdin.readline().split())

    remaining = N - (T + A)  # votes still uncounted

    # If one candidate leads by more votes than remain, the result is fixed
    if T > A + remaining or A > T + remaining:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()