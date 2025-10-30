import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

events = []
for i in range(M):
    T = int(data[2 + 3 * i])
    W = int(data[3 + 3 * i])
    S = int(data[4 + 3 * i])
    events.append((T, W, S))

# Sort events by time
events.sort()

# Initialize the current time and the list of people in the row
current_time = 0
people_in_row = list(range(1, N + 1))

# Initialize the list of noodles each person has
noodles = [0] * N

# Process each event
for T, W, S in events:
    # Update the current time
    current_time = T
    
    # Process all people in the row up to the current time
    while people_in_row and people_in_row[0] <= current_time:
        person = people_in_row.pop(0)
        noodles[person - 1] += W
    
    # Add the person who will return to the row
    people_in_row.append(person)
    
    # Update the current time to the return time
    current_time += S

# Output the total amount of noodles each person has
for noodle in noodles:
    print(noodle)