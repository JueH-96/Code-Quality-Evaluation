import sys

def solve(N, X, Y, Z):
    # Determine the direction of travel
    if X < Y:
        # Inbound train
        return Z > X and Z <= Y
    else:
        # Outbound train
        return Z < X and Z >= Y

def main():
    # Read input from stdin
    N, X, Y, Z = map(int, sys.stdin.readline().split())

    # Solve the problem
    result = solve(N, X, Y, Z)

    # Print the result to stdout
    print("Yes" if result else "No")

if __name__ == "__main__":
    main()