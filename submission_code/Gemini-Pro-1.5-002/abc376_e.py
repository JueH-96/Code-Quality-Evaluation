def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    pairs = []
    for i in range(n):
        pairs.append((a[i], b[i]))
    
    pairs.sort()

    min_val = float('inf')

    for i in range(n - k + 1):
        current_pairs = pairs[i:i+k]
        current_pairs.sort(key=lambda x: x[1])
        
        max_a = 0
        sum_b = 0
        
        for j in range(k):
            max_a = max(max_a, current_pairs[j][0])
            
        
        
        for j in range(k):
            sum_b += current_pairs[j][1]
        
        min_val = min(min_val, max_a * sum_b)
        
        for x in range(1 << n):
            if bin(x).count('1') == k:
                current_a = []
                current_b = []
                for j in range(n):
                    if (x >> j) & 1:
                        current_a.append(a[j])
                        current_b.append(b[j])
                
                max_a = max(current_a)
                sum_b = sum(current_b)
                min_val = min(min_val, max_a * sum_b)

    print(min_val)


t = int(input())
for _ in range(t):
    solve()