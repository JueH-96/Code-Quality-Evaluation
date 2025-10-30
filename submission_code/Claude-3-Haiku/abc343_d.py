from collections import defaultdict

N, T = map(int, input().split())
scores = [0] * N

for _ in range(T):
    A, B = map(int, input().split())
    scores[A-1] += B

result = []
for i in range(T):
    unique_scores = set()
    for score in scores:
        unique_scores.add(score)
    result.append(len(unique_scores))
    
    for j in range(N):
        scores[j] += B
        
print('
'.join(map(str, result)))