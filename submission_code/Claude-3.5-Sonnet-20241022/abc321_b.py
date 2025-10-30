N, X = map(int, input().split())
A = list(map(int, input().split()))

def get_final_grade(scores):
    scores.sort()
    return sum(scores[1:-1])

# Try each possible score for round N
for score in range(101):
    scores = A + [score]
    grade = get_final_grade(scores[:])
    if grade >= X:
        print(score)
        exit()

print(-1)