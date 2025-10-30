import itertools

def main():
    # Read input scores
    a, b, c, d, e = map(int, input().split())
    
    # Map letters to their corresponding scores
    letter_scores = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}
    
    # Generate all non-empty combinations of 'ABCDE'
    letters = 'ABCDE'
    participants = []
    for r in range(1, 6):
        for combo in itertools.combinations(letters, r):
            name = ''.join(combo)
            total_score = sum(letter_scores[letter] for letter in name)
            participants.append((total_score, name))
    
    # Sort participants by descending score, then ascending name
    participants_sorted = sorted(participants, key=lambda x: (-x[0], x[1]))
    
    # Print the sorted names
    for participant in participants_sorted:
        print(participant[1])

if __name__ == "__main__":
    main()