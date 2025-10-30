def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, T = map(int, data[0].split())
    scores = [0] * (N + 1)  # scores[0] is unused, scores[1] to scores[N] are for players 1 to N
    score_count = {}
    
    results = []
    
    for i in range(1, T + 1):
        A, B = map(int, data[i].split())
        
        # Update the score of player A
        current_score = scores[A]
        new_score = current_score + B
        scores[A] = new_score
        
        # Update the score count
        if current_score in score_count:
            score_count[current_score] -= 1
            if score_count[current_score] == 0:
                del score_count[current_score]
        
        if new_score in score_count:
            score_count[new_score] += 1
        else:
            score_count[new_score] = 1
        
        # The number of different scores is the size of the score_count dictionary
        results.append(len(score_count))
    
    # Print all results
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()