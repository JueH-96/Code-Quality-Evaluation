def main():
    import sys
    
    # Read the input scores for problems A, B, C, D, E.
    data = sys.stdin.read().strip().split()
    if not data:
        return
    a, b, c, d, e = map(int, data)
    
    # Map each problem letter to its corresponding score.
    score_map = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}
    problems = "ABCDE"
    
    # Generate all non-empty subsequences (there are 2^5 - 1 = 31 valid subsequences).
    participants = []
    for mask in range(1, 1 << 5):
        name = ""
        total_score = 0
        for i in range(5):
            if mask & (1 << i):
                letter = problems[i]
                name += letter
                total_score += score_map[letter]
        participants.append((total_score, name))
    
    # Sort participants by descending score, and for ties, sort lexicographically (as defined).
    # Python's native string comparison is identical to the dictionary order defined here.
    participants.sort(key=lambda x: (-x[0], x[1]))
    
    # Output the participant names in order.
    output = []
    for score, participant_name in participants:
        output.append(participant_name)
    
    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()