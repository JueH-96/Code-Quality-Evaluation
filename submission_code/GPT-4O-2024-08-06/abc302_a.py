# YOUR CODE HERE
import sys
import math

def minimum_attacks(A, B):
    # Calculate the minimum number of attacks needed
    return (A + B - 1) // B

if __name__ == "__main__":
    # Read input
    input_data = sys.stdin.read().strip()
    A, B = map(int, input_data.split())
    
    # Calculate and print the result
    print(minimum_attacks(A, B))