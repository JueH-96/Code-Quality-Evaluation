def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:]))

    # If there's only one person, they're trivially the strongest already.
    if N == 1:
        print(0)
        return

    # Person 1's score:
    p1 = P[0]
    # Maximum score among the others:
    max_other = max(P[1:])
    # We need p1 + x > max_other  =>  x > max_other - p1  =>  x = max(0, max_other - p1 + 1)
    needed = max(0, max_other - p1 + 1)
    print(needed)

if __name__ == "__main__":
    main()