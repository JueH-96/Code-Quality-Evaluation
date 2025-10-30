import sys

def main() -> None:
    # Read N and K
    N, K = map(int, sys.stdin.readline().split())
    
    # Read the list of card numbers (from top to bottom)
    A = list(map(int, sys.stdin.readline().split()))
    
    # Move the bottom K cards to the top while preserving order
    result = A[N - K:] + A[:N - K]
    
    # Output the resulting stack from top to bottom
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()