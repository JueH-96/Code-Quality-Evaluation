import sys

def main() -> None:
    # Read input
    data = sys.stdin.read().strip().split()
    
    # Extract N, M, P as integers
    N, M, P = map(int, data)
    
    # If the first observable full moon is after day N, answer is 0
    if M > N:
        print(0)
        return
    
    # Otherwise, count how many terms of the arithmetic progression
    # M, M+P, M+2P, ... do not exceed N.
    # General k-th term (0-indexed) is M + k*P ≤ N
    # Largest k  : k_max = (N - M) // P
    # Number of terms = k_max + 1
    full_moon_count = (N - M) // P + 1
    
    print(full_moon_count)

if __name__ == "__main__":
    main()