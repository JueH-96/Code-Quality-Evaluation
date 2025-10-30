# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    T = int(data[1])
    
    events = []
    for i in range(T):
        A = int(data[2 + 2 * i])
        B = int(data[3 + 2 * i])
        events.append((A, B))
    
    scores = [0] * N
    unique_scores = set([0])
    
    results = []
    for A, B in events:
        player_index = A - 1
        old_score = scores[player_index]
        new_score = old_score + B
        
        if old_score in unique_scores:
            unique_scores.remove(old_score)
        scores[player_index] = new_score
        unique_scores.add(new_score)
        
        results.append(len(unique_scores))
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()