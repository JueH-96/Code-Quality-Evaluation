def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    
    # Initialize a mapping from each character to itself
    mapping = {chr(ord('a') + i): chr(ord('a') + i) for i in range(26)}
    
    index = 3
    for _ in range(Q):
        c = data[index]
        d = data[index+1]
        index += 2
        # Update the mapping for all characters that currently map to c
        # We need to find all keys that map to c and update them to map to d
        # To do this efficiently, we can iterate over all keys and update if their current mapping is c
        for key in mapping:
            if mapping[key] == c:
                mapping[key] = d
    
    # Now, apply the mapping to the string S
    result = []
    for char in S:
        result.append(mapping[char])
    
    print(''.join(result))

if __name__ == "__main__":
    main()