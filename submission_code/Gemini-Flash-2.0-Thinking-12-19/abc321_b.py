def solve():
    n, x = map(int, input().split())
    a_scores = list(map(int, input().split()))
    min_score_needed = -1
    low = 0
    high = 100
    
    while low <= high:
        mid_score = (low + high) // 2
        current_scores = a_scores + [mid_score]
        current_scores.sort()
        current_grade = sum(current_scores[1:n-1])
        if current_grade >= x:
            min_score_needed = mid_score
            high = mid_score - 1
        else:
            low = mid_score + 1
            
    print(min_score_needed)

if __name__ == '__main__':
    solve()