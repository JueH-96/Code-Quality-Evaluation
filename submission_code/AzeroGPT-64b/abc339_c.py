def passengers_on_bus(N, stops):
    """
    Calculate the minimum possible number of passengers on a bus.
    
    Parameters:
    N (int): The number of stops.
    stops (list of int): The changes in the number of passengers at each stop.
    
    Returns:
    int: The minimum possible number of passengers on the bus.
    """
    passengers = 0
    min_passengers = 0
    for change in stops:
        passengers += change
        if passengers < min_passengers:
            passengers = min_passengers
        else:
            min_passengers = passengers
    return min_passengers

# Reading input from stdin
N = int(input())
stops = list(map(int, input().split()))

# Calculating and printing the answer
print(passengers_on_bus(N, stops))