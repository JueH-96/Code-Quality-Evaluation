def solve():
    n = int(input())
    a = list(map(int, input().split()))
    counts = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1
    
    condition1_pairs = 0
    for value in counts:
        count = counts[value]
        condition1_pairs += count * (count - 1) // 2
        
    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    condition2_pairs = c1 * c2
    
    result = condition1_pairs + condition2_pairs
    print(result)

t = int(input())
for _ in range(t):
    solve()