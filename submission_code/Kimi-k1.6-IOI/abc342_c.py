import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    q = int(sys.stdin.readline())
    
    # Initialize the mapping where each character maps to itself
    char_map = list(range(26))
    
    for _ in range(q):
        c, d = sys.stdin.readline().split()
        c_idx = ord(c) - ord('a')
        d_idx = ord(d) - ord('a')
        # Update all characters that currently map to c_idx to map to d_idx
        for x in range(26):
            if char_map[x] == c_idx:
                char_map[x] = d_idx
    
    # Build the result string
    result = []
    for char in s:
        original_idx = ord(char) - ord('a')
        new_char = chr(char_map[original_idx] + ord('a'))
        result.append(new_char)
    
    print(''.join(result))

if __name__ == "__main__":
    main()