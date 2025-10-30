# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

min_initial_passengers = 0  # Minimum initial number of passengers
current_sum = 0  # Running sum of passenger changes

for a in A:
    current_sum += a
    # If we have a negative number of passengers at any point,
    # we need to adjust the initial number upward
    min_initial_passengers = max(min_initial_passengers, -current_sum)

# Current number of passengers = initial + all changes
current_passengers = min_initial_passengers + sum(A)
print(current_passengers)