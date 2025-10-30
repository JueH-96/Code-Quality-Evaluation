def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    strings = data[1:]
    
    # Determine the maximum length of the strings
    M = max(len(s) for s in strings)
    
    # Create a list of lists to hold the characters in vertical format
    vertical = [['*'] * N for _ in range(M)]
    
    # Fill the vertical list with characters from the strings
    for i in range(N):
        s = strings[i]
        length = len(s)
        for j in range(length):
            vertical[j][N-i-1] = s[j]
    
    # Convert each list in vertical to a string and make sure it does not end with '*'
    result = []
    for line in vertical:
        # Convert list to string
        line_str = ''.join(line)
        # Strip trailing '*' characters
        line_str = line_str.rstrip('*')
        result.append(line_str)
    
    # Print the result
    for line in result:
        print(line)

if __name__ == "__main__":
    main()