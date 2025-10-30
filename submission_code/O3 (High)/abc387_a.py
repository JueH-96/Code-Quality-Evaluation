import sys

def main():
    # Read the two integers A and B
    a, b = map(int, sys.stdin.read().strip().split())
    
    # Compute (A + B)^2
    result = (a + b) ** 2
    
    # Output the result
    print(result)

# Ensure main() is executed
if __name__ == "__main__":
    main()