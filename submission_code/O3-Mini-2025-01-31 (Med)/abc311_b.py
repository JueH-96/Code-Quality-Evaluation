def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # First two are N and D
    N = int(input_data[0])
    D = int(input_data[1])
    schedules = input_data[2:]
    
    # Create a list representing if every person is free on day j
    # There are D days, index 0...D-1
    free_all = [True] * D
    for schedule in schedules:
        for j in range(D):
            if schedule[j] != 'o':
                free_all[j] = False
    
    max_consecutive = 0
    current = 0
    for j in range(D):
        if free_all[j]:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0

    print(max_consecutive)

if __name__ == '__main__':
    main()