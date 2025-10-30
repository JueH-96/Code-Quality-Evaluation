import sys

def solve(N, A, T):
    """
    Calculate the time when each person will finish purchasing their ticket.

    Args:
    N (int): The number of people.
    A (int): The time it takes to purchase a ticket.
    T (list): A list of arrival times for each person.

    Returns:
    list: A list of times when each person will finish purchasing their ticket.
    """
    finish_times = []
    current_time = 0

    for i in range(N):
        # If the current time is less than the arrival time, update the current time
        if current_time < T[i]:
            current_time = T[i]
        
        # Calculate the finish time for the current person
        finish_time = current_time + A
        
        # Update the current time for the next person
        current_time = finish_time
        
        # Append the finish time to the list
        finish_times.append(finish_time)

    return finish_times

def main():
    # Read the input from stdin
    N, A = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))

    # Calculate the finish times
    finish_times = solve(N, A, T)

    # Print the finish times
    for finish_time in finish_times:
        print(finish_time)

if __name__ == "__main__":
    main()