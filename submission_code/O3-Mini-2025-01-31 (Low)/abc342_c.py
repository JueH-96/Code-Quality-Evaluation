def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()
    Q = int(input().strip())
    operations = []
    for _ in range(Q):
        line = input().strip().split()
        operations.append((line[0], line[1]))
    
    # Create a mapping for each character 'a' to 'z', initially identity mapping.
    mapping = [chr(i + ord('a')) for i in range(26)]
    
    # Process operations in reverse order.
    for c, d in reversed(operations):
        # All occurrences of c should eventually be replaced by what d maps to.
        mapping[ord(c) - ord('a')] = mapping[ord(d) - ord('a')]
    
    # Build result string based on mapping.
    result = []
    for char in S:
        result.append(mapping[ord(char) - ord('a')])
    print("".join(result))

if __name__ == '__main__':
    main()