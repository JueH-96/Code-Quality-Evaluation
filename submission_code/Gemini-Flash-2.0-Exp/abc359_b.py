def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    for i in range(1, n + 1):
        indices = [idx for idx, val in enumerate(a) if val == i]
        if len(indices) == 2 and abs(indices[0] - indices[1]) == 2:
            count += 1
    
    print(count)

solve()