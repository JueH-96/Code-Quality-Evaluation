def main():
    import sys
    input = sys.stdin.readline

    N, T = map(int, input().split())
    # Initialize scores of all players
    scores = [0] * (N + 1)  # 1-indexed
    # Dictionary mapping score -> count
    score_counts = {0: N}
    
    # Process each update
    for _ in range(T):
        a, b = map(int, input().split())
        # Get the old score of player a
        old_score = scores[a]
        new_score = old_score + b
        scores[a] = new_score

        # Decrement the count for the old score
        score_counts[old_score] -= 1
        if score_counts[old_score] == 0:
            del score_counts[old_score]
            
        # Increment count for the new score
        if new_score in score_counts:
            score_counts[new_score] += 1
        else:
            score_counts[new_score] = 1
        
        # The number of distinct score values is the size of the dictionary
        sys.stdout.write(str(len(score_counts)) + "
")

if __name__ == '__main__':
    main()