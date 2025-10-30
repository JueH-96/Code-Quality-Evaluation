# YOUR CODE HERE
def solve():
    m = int(input())
    s = [input() for _ in range(3)]
    
    ans = float('inf')
    
    for target_digit in range(10):
        target_digit = str(target_digit)
        
        times = []
        for i in range(3):
            found = False
            for j in range(m):
                if s[i][j] == target_digit:
                    times.append(j)
                    found = True
                    break
            if not found:
                
                break
        else:
            
            
            perms = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
            for perm in perms:
                
                t1 = times[perm[0]]
                t2 = times[perm[1]]
                t3 = times[perm[2]]
                
                cur_time = 0
                
                cur_time = t1
                
                
                t2_time = (t2 - t1) % m
                if t2_time < 0:
                    t2_time += m
                cur_time = max(cur_time, t1 + t2_time)
                
                t3_time = (t3 - cur_time) % m
                if t3_time < 0:
                    t3_time += m
                cur_time = max(cur_time, cur_time + t3_time)
                
                ans = min(ans, cur_time)
                
    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()