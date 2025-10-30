# YOUR CODE HERE
import sys

def count_arithmetic_sequences(A, B):
    # Possible values of x to form an arithmetic sequence with A and B
    x1 = 2 * A - B
    x2 = 2 * B - A
    x3 = (A + B) // 2
    
    # Check if x3 is valid (A + B should be even for x3 to be an integer)
    if (A + B) % 2 == 0:
        # Use a set to avoid counting duplicates
        possible_x = {x1, x2, x3}
    else:
        possible_x = {x1, x2}
    
    return len(possible_x)

def main():
    input = sys.stdin.read().strip().split()
    A = int(input[0])
    B = int(input[1])
    result = count_arithmetic_sequences(A, B)
    print(result)

if __name__ == "__main__":
    main()