import sys

def main():
    # Read input from stdin
    input_sequence = sys.stdin.read().strip().split()

    # Convert the input sequence to a list of integers
    binary_sequence = [int(bit) for bit in input_sequence]

    # Calculate the result using the formula A_0 * 2^0 + A_1 * 2^1 + ... + A_63 * 2^63
    result = sum(bit * (2 ** index) for index, bit in enumerate(binary_sequence))

    # Print the result to stdout
    print(result)

if __name__ == "__main__":
    main()