def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    s = [input() for _ in range(n)]

    for i in range(n):
        scores = []
        for j in range(n):
            score = j + 1
            for k in range(m):
                if s[j][k] == 'o':
                    score += a[k]
            scores.append(score)

        my_score = i + 1
        for k in range(m):
            if s[i][k] == 'o':
                my_score += a[k]

        
        greater_scores = [score for idx, score in enumerate(scores) if idx != i and score >= my_score]
        
        if not greater_scores:
            print(0)
            continue
        
        
        unsolved_problems = []
        for k in range(m):
            if s[i][k] == 'x':
                unsolved_problems.append(a[k])
        
        unsolved_problems.sort(reverse=True)
        
        ans = 0
        temp_score = my_score
        
        greater_scores = [score for idx, score in enumerate(scores) if idx != i]
        
        while any(temp_score <= other_score for other_score in greater_scores):
            temp_score += unsolved_problems[ans]
            ans += 1
        
        print(ans)

solve()