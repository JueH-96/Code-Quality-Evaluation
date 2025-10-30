N, X = map(int, input().split())
A = list(map(int, input().split()))

answer = -1
for s_n in range(101):  # Try scores 0 to 100 for round N
    scores = A + [s_n]  # Add the candidate score
    scores.sort()  # Sort all scores
    final_grade = sum(scores[1:-1])  # Sum excluding first (lowest) and last (highest)
    if final_grade >= X:
        answer = s_n
        break

print(answer)