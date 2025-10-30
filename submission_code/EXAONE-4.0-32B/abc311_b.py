def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
        
    n, d = map(int, data[0].split())
    schedules = []
    for i in range(1, 1 + n):
        schedules.append(data[i].strip())
    
    free_day = [True] * d
    
    for j in range(d):
        for i in range(n):
            if schedules[i][j] == 'x':
                free_day[j] = False
                break
    
    current = 0
    max_streak = 0
    for j in range(d):
        if free_day[j]:
            current += 1
        else:
            if current > max_streak:
                max_streak = current
            current = 0
            
    if current > max_streak:
        max_streak = current
        
    print(max_streak)

if __name__ == "__main__":
    main()