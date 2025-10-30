def main():
    import sys
    input = sys.stdin.readline

    # Read the first line: number of people (N) and time to purchase ticket (A)
    N, A = map(int, input().split())
    
    # Read the arrival times
    T = list(map(int, input().split()))
    
    # current_time will record when the ticket booth becomes free.
    current_time = 0

    results = []  # list to store finish times for each person
    
    for t in T:
        # The person starts purchasing once current_time or their arrival time, whichever is later.
        start_time = max(t, current_time)
        finish_time = start_time + A  # finish time for current person
        results.append(finish_time)
        # Update the current_time to this finish time
        current_time = finish_time
    
    # Output each finish time.
    print("
".join(map(str, results)))
    
if __name__ == '__main__':
    main()