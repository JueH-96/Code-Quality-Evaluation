# YOUR CODE HERE
n, t = map(int, input().split())

# Initialize scores for all players
scores = [0] * (n + 1)  # 1-indexed, so we need n+1 elements

for _ in range(t):
    a, b = map(int, input().split())
    # Update player a's score
    scores[a] += b
    
    # Count distinct scores among players 1 to n
    distinct_scores = set(scores[1:n+1])
    print(len(distinct_scores))