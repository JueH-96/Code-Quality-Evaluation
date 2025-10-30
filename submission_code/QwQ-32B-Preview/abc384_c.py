def main():
    # Read input scores for problems A, B, C, D, E
    scores_input = input().split()
    scores = [int(scores_input[0]), int(scores_input[1]), int(scores_input[2]), int(scores_input[3]), int(scores_input[4])]
    
    # Define the problems corresponding to each bit in the mask
    problems = "ABCDE"
    
    # List to store (score, name) tuples
    participants = []
    
    # Generate all non-empty subsequences using bit masks
    for mask in range(1, 32):  # 31 participants (1 to 31 inclusive)
        name = ''
        total_score = 0
        for i in range(5):
            if mask & (1 << i):
                name += problems[i]
                total_score += scores[i]
        participants.append((total_score, name))
    
    # Sort the participants: descending by score, ascending by name
    participants_sorted = sorted(participants, key=lambda x: (-x[0], x[1]))
    
    # Print the sorted names
    for participant in participants_sorted:
        print(participant[1])

if __name__ == "__main__":
    main()