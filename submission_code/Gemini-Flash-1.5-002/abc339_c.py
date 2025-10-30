# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

current_passengers = 0
min_passengers = 0
for passenger_change in a:
    current_passengers += passenger_change
    min_passengers = min(min_passengers, current_passengers)

print(current_passengers - min_passengers)