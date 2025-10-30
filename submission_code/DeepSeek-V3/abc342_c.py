# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    
    operations = []
    index = 3
    for _ in range(Q):
        c = data[index]
        d = data[index+1]
        operations.append((c, d))
        index += 2
    
    # Initialize a mapping from each character to itself
    mapping = {chr(ord('a') + i): chr(ord('a') + i) for i in range(26)}
    
    for c, d in operations:
        # Update the mapping for all characters that currently map to c
        for key in mapping:
            if mapping[key] == c:
                mapping[key] = d
    
    # Apply the final mapping to the string S
    result = []
    for char in S:
        result.append(mapping[char])
    
    print(''.join(result))

if __name__ == "__main__":
    main()