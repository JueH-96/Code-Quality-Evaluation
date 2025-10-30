def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))

    min_s = -1

    for s in range(101):
        all_scores = sorted(a + [s])
        if n >= 3:
            final_grade = sum(all_scores[1:n-1])
            if final_grade >= x:
                if min_s == -1:
                    min_s = s
                else:
                    min_s = min(min_s, s)

    print(min_s)

solve()