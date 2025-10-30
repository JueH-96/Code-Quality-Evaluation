# YOUR CODE HERE
def count_trailing_zeros(n):
    # Convert to binary and strip the '0b' prefix
    binary_representation = bin(n)[2:]
    # Reverse the binary string to count trailing zeros
    reversed_binary = binary_representation[::-1]
    # Count the number of zeros until we hit a '1'
    count = 0
    for char in reversed_binary:
        if char == '0':
            count += 1
        else:
            break
    return count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    print(count_trailing_zeros(N))