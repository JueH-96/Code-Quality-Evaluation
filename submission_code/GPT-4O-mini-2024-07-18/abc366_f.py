from itertools import permutations
import sys

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and K
    N, K = map(int, data[0].split())
    
    # Read the coefficients A and B
    functions = [tuple(map(int, line.split())) for line in data[1:N+1]]
    
    max_value = float('-inf')
    
    # Generate all permutations of K distinct indices from 0 to N-1
    for indices in permutations(range(N), K):
        # Start with x = 1
        x = 1
        # Apply the functions in the order of the current permutation
        for index in indices:
            A, B = functions[index]
            x = A * x + B
        # Update the maximum value found
        max_value = max(max_value, x)
    
    print(max_value)

if __name__ == "__main__":
    main()