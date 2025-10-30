from collections import Counter

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    
    count = Counter(a)
    
    result = 0
    
    # Count pairs where both elements have the same value
    for k in count.values():
        result += k * (k - 1) // 2
    
    # Count pairs where one element is 1 and the other is 2
    if 1 in count and 2 in count:
        result += count[1] * count[2]
    
    print(result)