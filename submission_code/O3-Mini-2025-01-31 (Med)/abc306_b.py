def main():
    import sys
    # Read the input from stdin and split into tokens.
    tokens = sys.stdin.read().strip().split()
    # Convert each token to int.
    bits = list(map(int, tokens))
    
    # Compute the sum A_0*2^0 + A_1*2^1 + ... + A_{63}*2^63.
    result = sum(bit << i for i, bit in enumerate(bits))
    
    # Output the result.
    print(result)

if __name__ == '__main__':
    main()