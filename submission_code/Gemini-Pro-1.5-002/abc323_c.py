# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    s = [input() for _ in range(n)]

    for i in range(n):
        current_score = i + 1
        for j in range(m):
            if s[i][j] == 'o':
                current_score += a[j]
        
        ans = 0
        
        
        scores = []
        for k in range(n):
            score = k + 1
            for j in range(m):
                if s[k][j] == 'o':
                    score += a[j]
            scores.append(score)

        
        
        unsolved = []
        for j in range(m):
            if s[i][j] == 'x':
                unsolved.append(a[j])
        
        unsolved.sort(reverse=True)
        
        count = 0
        
        while True:
            
            max_other_score = 0
            for k in range(n):
                if k != i:
                    max_other_score = max(max_other_score, scores[k])
            
            if current_score > max_other_score:
                print(count)
                break
            else:
                if count < len(unsolved):
                    current_score += unsolved[count]
                    count += 1
                else:
                    print(count)
                    break
solve()