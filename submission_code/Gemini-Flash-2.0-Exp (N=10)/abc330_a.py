def solve():
    n, l = map(int, input().split())
    a = list(map(int, input().split()))
    
    passed_count = 0
    for score in a:
        if score >= l:
            passed_count += 1
    
    print(passed_count)

solve()