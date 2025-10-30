def main():
    import sys
    # Read all input as list of strings (tokens)
    tokens = sys.stdin.read().split()
    # Convert tokens to integers (0 or 1)
    bits = [int(token) for token in tokens]
    # Calculate the sum: A_0*2^0 + A_1*2^1 + ... + A_63*2^63
    # The list bits is indexed from 0 to 63
    result = 0
    for i, bit in enumerate(bits):
        if bit:
            result += 1 << i  # Using bit shift for efficiency
    # Print the result
    print(result)

if __name__ == '__main__':
    main()