n = int(input())
scores = list(map(int, input().split()))

ranks = [0] * n
unique_scores = sorted(set(scores), reverse=True)
r = 1

for score in unique_scores:
    # Find all people with this score
    people_with_score = [i for i in range(n) if scores[i] == score]
    
    # Assign rank r to all these people
    for idx in people_with_score:
        ranks[idx] = r
    
    # Update r
    r += len(people_with_score)

# Print the ranks
for rank in ranks:
    print(rank)