def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = list(data[1])
    Q = int(data[2])
    
    operations = data[3:]
    
    # To handle multiple operations efficiently, we use a mapping strategy
    # where we map each character to its final character after all operations
    # This avoids multiple passes through the string S
    
    # Initialize the mapping of each character to itself
    char_map = {chr(i): chr(i) for i in range(ord('a'), ord('z') + 1)}
    
    # Process each operation and update the mapping
    for i in range(Q):
        c_i = operations[2 * i]
        d_i = operations[2 * i + 1]
        
        # Update all keys in char_map that currently map to c_i to map to d_i
        for key in list(char_map.keys()):
            if char_map[key] == c_i:
                char_map[key] = d_i
    
    # Apply the final mapping to the string S
    result = ''.join(char_map[ch] for ch in S)
    
    # Print the final string
    print(result)

if __name__ == "__main__":
    main()