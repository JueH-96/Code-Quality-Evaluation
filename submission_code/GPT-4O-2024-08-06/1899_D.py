def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index + n]))
        index += n
        
        from collections import Counter
        count = Counter(a)
        
        total_pairs = 0
        for value in count.values():
            if value > 1:
                total_pairs += value * (value - 1) // 2
        
        results.append(total_pairs)
    
    for result in results:
        print(result)