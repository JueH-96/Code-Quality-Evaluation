def main():
    import sys
    
    # Read the bits as integers from stdin
    bits = list(map(int, sys.stdin.read().split()))
    
    # Calculate the result using summation of Ai * 2^i
    result = 0
    for i, bit in enumerate(bits):
        result += bit * (1 << i)
    
    # Output the result
    print(result)

# Do not remove or rename this call
main()