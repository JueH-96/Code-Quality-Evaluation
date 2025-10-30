def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    L = int(data[1])
    scores = list(map(int, data[2:]))

    count_passed = sum(score >= L for score in scores)
    print(count_passed)

# Call solve()    
solve()