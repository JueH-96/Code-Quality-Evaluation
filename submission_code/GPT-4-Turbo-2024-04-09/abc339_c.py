def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # We need to find the minimum possible current number of passengers
    # that is consistent with the given information.
    
    # Calculate the cumulative sum of A to simulate the passenger changes
    cumulative_sum = 0
    min_cumulative_sum = 0  # This will track the minimum cumulative sum
    
    for change in A:
        cumulative_sum += change
        if cumulative_sum < min_cumulative_sum:
            min_cumulative_sum = cumulative_sum
    
    # The minimum number of passengers at the start to ensure non-negativity
    # at all times would be the absolute value of the minimum cumulative sum
    # if it's negative, otherwise it's 0.
    min_initial_passengers = max(0, -min_cumulative_sum)
    
    # The current number of passengers would be this minimum initial number
    # plus the total change in passengers.
    current_passengers = min_initial_passengers + cumulative_sum
    
    print(current_passengers)

if __name__ == "__main__":
    main()