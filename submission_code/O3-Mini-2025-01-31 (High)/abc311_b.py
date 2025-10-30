def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    # Read N and D
    N = int(data[0])
    D = int(data[1])
    
    # Read the schedules for each person
    schedules = data[2:]
    
    max_consecutive = 0  # Maximum consecutive free days across all persons
    current_consecutive = 0  # Current consecutive free days count
    
    # Iterate over each day (0-indexed)
    for day in range(D):
        all_free = True
        for i in range(N):
            if schedules[i][day] != 'o':
                all_free = False
                break
        if all_free:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    print(max_consecutive)

if __name__ == "__main__":
    main()