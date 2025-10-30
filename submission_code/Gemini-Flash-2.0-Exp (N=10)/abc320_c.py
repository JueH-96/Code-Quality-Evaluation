def solve():
    m = int(input())
    s1 = input()
    s2 = input()
    s3 = input()

    ans = float('inf')

    for char_idx in range(m):
        char = s1[char_idx]
        
        times1 = [char_idx]
        
        times2 = []
        for i in range(m):
            if s2[i] == char:
                times2.append(i)
        
        times3 = []
        for i in range(m):
            if s3[i] == char:
                times3.append(i)
        
        if not times2 or not times3:
            continue
        
        for t1 in times1:
            for t2 in times2:
                for t3 in times3:
                    max_time = max(t1, t2, t3)
                    ans = min(ans, max_time)
    
    if ans == float('inf'):
        print("-1")
    else:
        print(ans)

solve()