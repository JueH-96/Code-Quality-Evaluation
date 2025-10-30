def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))

    for new_score in range(101):
        scores = sorted(a + [new_score])
        final_grade = sum(scores[1:n-1])
        if final_grade >= x:
            print(new_score)
            return
    print("-1")

solve()