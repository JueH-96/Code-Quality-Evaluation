def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()
    Q = int(input().strip())

    # We create a mapping for each letter from 'a' to 'z'.
    # Initially, each letter maps to itself.
    mapping = [chr(i + ord('a')) for i in range(26)]

    # For each operation, we update the current mapping:
    # Operation "c d" replaces every occurrence of letter c with letter d.
    # In our mapping, if a letter's mapped value is c, then after the operation
    # it should be updated to d.
    for _ in range(Q):
        line = input().split()
        if not line:
            continue
        c, d = line
        for i in range(26):
            if mapping[i] == c:
                mapping[i] = d

    # With the final mapping ready, convert each character in S accordingly.
    result = []
    for ch in S:
        result.append(mapping[ord(ch) - ord('a')])
    
    sys.stdout.write("".join(result))


if __name__ == '__main__':
    main()