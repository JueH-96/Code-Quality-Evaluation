def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    schedules = data[2:]
    
    # Create a list to track which days all people are free
    all_free = [True] * D
    
    # Iterate over each person's schedule
    for schedule in schedules:
        for day in range(D):
            if schedule[day] == 'x':
                all_free[day] = False
    
    # Find the longest sequence of consecutive True values in all_free
    max_consecutive_free = 0
    current_free = 0
    
    for day in range(D):
        if all_free[day]:
            current_free += 1
            max_consecutive_free = max(max_consecutive_free, current_free)
        else:
            current_free = 0
    
    print(max_consecutive_free)

if __name__ == "__main__":
    main()