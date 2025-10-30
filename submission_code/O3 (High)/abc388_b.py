import sys

def main() -> None:
    # Read N and D
    N, D = map(int, sys.stdin.readline().split())

    # Read thicknesses and initial lengths
    snakes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    # For each possible increase k (1 … D), find the heaviest snake
    for k in range(1, D + 1):
        heaviest = 0
        for t, l in snakes:
            weight = t * (l + k)
            if weight > heaviest:
                heaviest = weight
        print(heaviest)

if __name__ == "__main__":
    main()