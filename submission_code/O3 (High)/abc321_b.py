import sys

def main() -> None:
    # Read every integer from stdin
    data = list(map(int, sys.stdin.read().split()))
    if not data:
        return

    N, X = data[0], data[1]           # number of rounds, required final grade
    A = data[2:]                      # scores of the first N-1 rounds

    sumA  = sum(A)                    # sum of known scores
    minA  = min(A)                    # current minimum
    maxA  = max(A)                    # current maximum

    # Try every possible score for the last round (0‥100)
    for s in range(101):
        # Final grade = (sum of all scores) - (smallest score) - (largest score)
        grade = (sumA + s) - min(minA, s) - max(maxA, s)
        if grade >= X:
            print(s)
            return

    # Impossible to reach X
    print(-1)

if __name__ == "__main__":
    main()