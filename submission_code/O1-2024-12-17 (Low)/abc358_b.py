def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, A = map(int, data[:2])
    T = list(map(int, data[2:]))

    finishing_times = []
    current_finish = 0
    
    for arrival_time in T:
        start_time = max(current_finish, arrival_time)
        finish_time = start_time + A
        finishing_times.append(finish_time)
        current_finish = finish_time
    
    print('
'.join(map(str, finishing_times)))

# Call main function
main()