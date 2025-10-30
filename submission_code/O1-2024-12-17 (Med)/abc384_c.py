def main():
    import sys
    
    # Read inputs
    a, b, c, d, e = map(int, sys.stdin.readline().split())
    
    # Map problem letters to their scores
    scores = {
        'A': a,
        'B': b,
        'C': c,
        'D': d,
        'E': e
    }
    
    # All non-empty subsets correspond to participants
    # We use a bit representation from 1 to 31 (inclusive) for subsets
    participants = []
    letters = ['A', 'B', 'C', 'D', 'E']
    
    for mask in range(1, 1 << 5):  # 1 to 31
        name = []
        total_score = 0
        for i in range(5):
            if mask & (1 << i):
                letter = letters[i]
                name.append(letter)
                total_score += scores[letter]
        participant_name = "".join(name)
        participants.append((participant_name, total_score))
    
    # Sort participants by descending score, then lexicographically by name
    participants.sort(key=lambda x: (-x[1], x[0]))
    
    # Print the sorted participant names
    for p in participants:
        print(p[0])

# Call main function
main()