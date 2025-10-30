def solve():
    n = int(input())
    a = list(map(int, input().split()))
    counts = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1
    
    result = 0
    for value in counts:
        count = counts[value]
        if count >= 2:
            result += count * (count - 1) // 2
            
    count1 = counts.get(1, 0)
    count2 = counts.get(2, 0)
    result += count1 * count2
    
    print(result)

t = int(input())
for _ in range(t):
    solve()