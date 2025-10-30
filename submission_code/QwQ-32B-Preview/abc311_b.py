def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and D
    N_D = data[0].split()
    N = int(N_D[0])
    D = int(N_D[1])
    
    # Read the schedules
    schedules = []
    for i in range(1, N+1):
        schedules.append(data[i])
    
    max_streak = 0
    current_streak = 0
    
    for day in range(D):
        all_free = True
        for person in range(N):
            if schedules[person][day] != 'o':
                all_free = False
                break
        if all_free:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    
    print(max_streak)

if __name__ == "__main__":
    main()