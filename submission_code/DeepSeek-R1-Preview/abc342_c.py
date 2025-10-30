import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    q = int(sys.stdin.readline())
    
    # Initialize mapping: each character maps to itself
    mapping = list('abcdefghijklmnopqrstuvwxyz')
    
    for _ in range(q):
        c, d = sys.stdin.readline().split()
        for i in range(26):
            if mapping[i] == c:
                mapping[i] = d
    
    # Build the result string
    result = []
    for char in s:
        result.append(mapping[ord(char) - ord('a')])
    
    print(''.join(result))

if __name__ == "__main__":
    main()