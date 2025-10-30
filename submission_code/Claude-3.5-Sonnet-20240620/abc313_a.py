# YOUR CODE HERE
N = int(input())
scores = list(map(int, input().split()))

max_score = max(scores[1:])  # Find the maximum score excluding person 1
person_1_score = scores[0]   # Score of person 1

if person_1_score > max_score:
    print(0)  # Person 1 is already the strongest
else:
    points_needed = max_score - person_1_score + 1
    print(points_needed)