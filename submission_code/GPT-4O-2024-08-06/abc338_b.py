# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Dictionary to count frequency of each character
    frequency = {}
    
    # Count the frequency of each character in the string
    for char in S:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    
    # Find the character with the maximum frequency
    max_freq = 0
    result_char = None
    
    for char in sorted(frequency.keys()):
        if frequency[char] > max_freq:
            max_freq = frequency[char]
            result_char = char
    
    # Print the result character
    print(result_char)

if __name__ == "__main__":
    main()