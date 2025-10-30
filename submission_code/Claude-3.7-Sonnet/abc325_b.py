def solve():
    N = int(input())
    
    # Count the number of employees who can attend for each UTC hour
    attendance = [0] * 24
    
    for _ in range(N):
        W, X = map(int, input().split())
        
        for hour in range(24):
            local_time = (X + hour) % 24
            if 9 <= local_time <= 17:
                attendance[hour] += W
    
    return max(attendance)

print(solve())