import sys
from collections import defaultdict

def main():
    n, t = map(int, sys.stdin.readline().split())
    current_scores = [0] * (n + 1)  # Using 1-based indexing
    frequency_map = defaultdict(int)
    frequency_map[0] = n  # Initially, all players have 0 points

    for _ in range(t):
        a, b = map(int, sys.stdin.readline().split())
        old_score = current_scores[a]
        # Update frequency map for the old score
        frequency_map[old_score] -= 1
        if frequency_map[old_score] == 0:
            del frequency_map[old_score]
        # Update the player's score
        current_scores[a] += b
        new_score = current_scores[a]
        # Update frequency map for the new score
        frequency_map[new_score] += 1
        # Output the number of unique scores
        print(len(frequency_map))

if __name__ == '__main__':
    main()