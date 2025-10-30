def solve():
    n, q = map(int, input().split())
    r = list(map(int, input().split()))
    
    for _ in range(q):
        x = int(input())
        
        sorted_r = sorted(r)
        
        count = 0
        total_reindeer = 0
        
        for reindeer_needed in sorted_r:
            if total_reindeer + reindeer_needed <= x:
                total_reindeer += reindeer_needed
                count += 1
            else:
                break
        
        print(count)

solve()