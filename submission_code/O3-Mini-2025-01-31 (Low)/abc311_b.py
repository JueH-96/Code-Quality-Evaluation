def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    D = int(input_data[1])
    schedules = input_data[2:]
    
    # Create a list where each index corresponds to a day: True if all persons are free, False otherwise.
    all_free_days = [True] * D
    for person in range(N):
        schedule = schedules[person]
        for day in range(D):
            if schedule[day] != 'o':
                all_free_days[day] = False
    
    # Find the maximum length of consecutive True values.
    max_consecutive = 0
    current_count = 0
    for day in range(D):
        if all_free_days[day]:
            current_count += 1
            if current_count > max_consecutive:
                max_consecutive = current_count
        else:
            current_count = 0
    
    print(max_consecutive)

if __name__ == '__main__':
    main()