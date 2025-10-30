def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))

    # Try every possible score x in [0, 100]
    for x in range(0, 101):
        B = A + [x]
        B.sort()
        # Final grade is sum of all except the smallest and the largest
        total = sum(B) - B[0] - B[-1]
        if total >= X:
            print(x)
            return

    # If no x works, print -1
    print(-1)

if __name__ == "__main__":
    main()