import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    T = int(data[1])
    
    events = []
    index = 2
    for _ in range(T):
        A_i = int(data[index]) - 1  # Convert to 0-based index
        B_i = int(data[index + 1])
        events.append((A_i, B_i))
        index += 2
    
    scores = [0] * N
    score_counts = {0: N}  # Initially, all players have 0 points
    
    results = []
    
    for player, points in events:
        old_score = scores[player]
        new_score = old_score + points
        
        # Update the player's score
        scores[player] = new_score
        
        # Update the score counts
        if old_score in score_counts:
            if score_counts[old_score] == 1:
                del score_counts[old_score]
            else:
                score_counts[old_score] -= 1
        
        if new_score in score_counts:
            score_counts[new_score] += 1
        else:
            score_counts[new_score] = 1
        
        # Number of different scores is the number of keys in score_counts
        results.append(len(score_counts))
    
    # Print results
    for result in results:
        print(result)

main()