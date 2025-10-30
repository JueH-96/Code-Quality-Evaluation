# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = int(data[1])
    T = list(map(int, data[2:]))
    
    # Initialize the current time when the booth will be free
    current_time = 0
    
    # Result list to store the finish times
    finish_times = []
    
    for i in range(N):
        arrival_time = T[i]
        
        # If the current time is less than the arrival time, the person can start immediately
        if current_time < arrival_time:
            current_time = arrival_time
        
        # The person will finish A seconds after they start
        finish_time = current_time + A
        finish_times.append(finish_time)
        
        # Update the current time to reflect when the booth will be free
        current_time = finish_time
    
    # Print the finish times for each person
    for finish_time in finish_times:
        print(finish_time)

if __name__ == "__main__":
    main()