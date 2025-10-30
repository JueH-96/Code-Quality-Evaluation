def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    from collections import Counter
    count = Counter(a)
    
    result = 0
    
    # Handle the special case where f(1) = f(2)
    count_1 = count.get(1, 0)
    count_2 = count.get(2, 0)
    result += count_1 * count_2
    
    # Handle identical values (including 1s with 1s and 2s with 2s)
    for val, cnt in count.items():
        if cnt > 1:
            result += cnt * (cnt - 1) // 2
    
    return result

t = int(input())
for _ in range(t):
    print(solve())