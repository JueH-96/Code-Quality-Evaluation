# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    schedules = data[2:2+N]
    
    # Create a list to mark if all are free on each day
    all_free = [True] * D
    
    for day in range(D):
        for person in range(N):
            if schedules[person][day] == 'x':
                all_free[day] = False
                break
    
    max_length = 0
    current_length = 0
    
    for day in range(D):
        if all_free[day]:
            current_length += 1
            if current_length > max_length:
                max_length = current_length
        else:
            current_length = 0
    
    print(max_length)

if __name__ == "__main__":
    main()