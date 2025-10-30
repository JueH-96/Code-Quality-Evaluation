def main():
    import sys
    from itertools import combinations
    input_data = sys.stdin.read().strip().split()
    a, b, c, d, e = map(int, input_data)
    score_map = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}
    problems = "ABCDE"
    
    participants = []
    # Generate all non-empty subsequences of problems (each as sorted order of letters)
    for r in range(1, 6):
        for comb in combinations(problems, r):
            name = ''.join(comb)
            score = sum(score_map[ch] for ch in name)
            participants.append((score, name))
    
    # Sort participants: descending score, and if scores equal, lexicographically (according to problem statement)
    participants.sort(key=lambda x: (-x[0], x[1]))
    
    # Output the names line by line
    out = []
    for score, name in participants:
        out.append(name)
    sys.stdout.write("
".join(out))
    
if __name__ == '__main__':
    main()