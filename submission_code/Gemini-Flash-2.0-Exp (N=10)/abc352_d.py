def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    
    min_diff = float('inf')
    
    for i in range(1 << n):
        indices = []
        for j in range(n):
            if (i >> j) & 1:
                indices.append(j)
        
        if len(indices) != k:
            continue
            
        subsequence = [p[idx] for idx in indices]
        subsequence.sort()
        
        is_good = True
        if k > 0:
            first = subsequence[0]
            for j in range(1, k):
                if subsequence[j] != first + j:
                    is_good = False
                    break
        else:
            is_good = False
        
        if is_good:
            min_diff = min(min_diff, indices[-1] - indices[0])
            
    print(min_diff)

solve()