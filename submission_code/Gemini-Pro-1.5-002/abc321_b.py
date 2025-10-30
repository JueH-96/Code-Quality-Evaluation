# YOUR CODE HERE
def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))

    def calculate_grade(scores):
        scores.sort()
        return sum(scores[1:-1])

    min_score = -1
    for score_n in range(101):
        scores = a + [score_n]
        grade = calculate_grade(scores)
        if grade >= x:
            min_score = score_n
            break
    
    print(min_score)

solve()