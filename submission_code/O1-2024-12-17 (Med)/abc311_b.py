def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, D = map(int, data[:2])
    schedules = data[2:]
    
    # Construct a list indicating for each day if all persons are free
    all_free = []
    for day in range(D):
        free_today = True
        for i in range(N):
            if schedules[i][day] == 'x':
                free_today = False
                break
        all_free.append(free_today)
    
    # Find the maximum length of consecutive True in all_free
    max_consecutive = 0
    current_consecutive = 0
    for day_free in all_free:
        if day_free:
            current_consecutive += 1
        else:
            current_consecutive = 0
        max_consecutive = max(max_consecutive, current_consecutive)
    
    print(max_consecutive)

# Call main function
if __name__ == "__main__":
    main()