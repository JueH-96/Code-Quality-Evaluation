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
    
    all_free = [True] * d
    for j in range(d):
        for i in range(n):
            if schedules[i][j] == 'x':
                all_free[j] = False
                break
                
    max_consecutive = 0
    current = 0
    for j in range(d):
        if all_free[j]:
            current += 1
        else:
            if current > max_consecutive:
                max_consecutive = current
            current = 0
    if current > max_consecutive:
        max_consecutive = current
        
    print(max_consecutive)

if __name__ == "__main__":
    main()