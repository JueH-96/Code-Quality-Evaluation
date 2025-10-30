def main():
    import sys
    data = sys.stdin.read().split()
    a, b, c, d, e = map(int, data)
    weights = [a, b, c, d, e]
    participants = []
    
    # There are 2^5 - 1 = 31 non-empty subsequences of "ABCDE"
    for mask in range(1, 1 << 5):
        name_chars = []
        score = 0
        for i in range(5):
            if (mask >> i) & 1:
                name_chars.append(chr(ord('A') + i))
                score += weights[i]
        name = ''.join(name_chars)
        participants.append(( -score, name ))
    
    # Sort by descending score (hence -score) then lexicographically by name
    participants.sort()
    
    # Output just the names, in order
    out = []
    for _, name in participants:
        out.append(name)
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()