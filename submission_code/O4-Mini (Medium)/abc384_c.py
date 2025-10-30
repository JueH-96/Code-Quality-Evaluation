def main():
    import sys
    a, b, c, d, e = map(int, sys.stdin.readline().split())
    scores = [a, b, c, d, e]
    letters = ['A', 'B', 'C', 'D', 'E']
    
    participants = []
    # Enumerate all non-empty subsets of {0,1,2,3,4}
    for mask in range(1, 1 << 5):
        name_chars = []
        total = 0
        for i in range(5):
            if mask & (1 << i):
                name_chars.append(letters[i])
                total += scores[i]
        name = ''.join(name_chars)
        participants.append((total, name))
    
    # Sort by descending score, then ascending name
    participants.sort(key=lambda x: (-x[0], x[1]))
    
    # Print only the names
    out = '
'.join(name for _, name in participants)
    print(out)

if __name__ == "__main__":
    main()