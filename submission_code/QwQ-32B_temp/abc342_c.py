import sys

def main():
    import sys
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    q = int(sys.stdin.readline())
    
    trans = list(range(26))  # Initialize each character to map to itself
    
    for _ in range(q):
        c, d = sys.stdin.readline().split()
        c_code = ord(c) - ord('a')
        d_code = ord(d) - ord('a')
        # Create new_trans based on current trans
        new_trans = [d_code if (trans[i] == c_code) else trans[i] for i in range(26)]
        trans = new_trans
    
    # Build the result string
    result = []
    for char in s:
        original = ord(char) - ord('a')
        mapped = trans[original]
        result.append(chr(mapped + ord('a')))
    
    print(''.join(result))

if __name__ == "__main__":
    main()