def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    operations = data[3: 3 + 2 * Q]
    
    # Initialize mapping where each character maps to itself
    mapping = [chr(i + ord('a')) for i in range(26)]
    
    # Apply operations in reverse order
    for i in range(Q-1, -1, -1):
        c = operations[2*i]
        d = operations[2*i + 1]
        mapping[ord(c) - ord('a')] = d
    
    # Build the new string using the mapping
    new_S = ''.join(mapping[ord(char) - ord('a')] for char in S)
    
    # Print the result
    print(new_S)

if __name__ == "__main__":
    main()