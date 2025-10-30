def main():
    import sys
    data = sys.stdin.read().strip().split()
    a, b, c, d, e = map(int, data)
    scores = [a, b, c, d, e]
    letters = ['A', 'B', 'C', 'D', 'E']
    
    participants = []
    # generate all non-empty subsets of ABCDE
    for mask in range(1, 1 << 5):
        name = []
        total = 0
        for i in range(5):
            if mask & (1 << i):
                name.append(letters[i])
                total += scores[i]
        pname = ''.join(name)
        participants.append((pname, total))
    
    # sort by descending score, then lexicographically by name
    participants.sort(key=lambda x: (-x[1], x[0]))
    
    out = []
    for pname, _ in participants:
        out.append(pname)
    sys.stdout.write("
".join(out))

main()