def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    
    for s_n in range(101):
        current_scores = a + [s_n]
        current_scores.sort()
        final_grade = sum(current_scores[1:n-1])
        if final_grade >= x:
            print(s_n)
            return
            
    print("-1")

if __name__ == '__main__':
    solve()