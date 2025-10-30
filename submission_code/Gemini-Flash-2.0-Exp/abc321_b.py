def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))

    for score_n in range(101):
        scores = sorted(a + [score_n])
        final_grade = sum(scores[1:n-1])
        if final_grade >= x:
            print(score_n)
            return

    print("-1")

solve()