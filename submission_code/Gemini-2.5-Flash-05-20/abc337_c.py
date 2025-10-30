import sys

def solve():
    N = int(sys.stdin.readline())
    # Read the line of integers as strings, then convert to a list of integers
    A_list = [int(x) for x in sys.stdin.readline().split()]

    # next_person[P] will store the person Q such that Q is right behind P.
    # We use 1-based indexing for people numbers (1 to N) as in the problem description.
    next_person = {}
    front_person = -1

    # Populate the next_person map and find the front_person.
    # A_list[i] describes person (i+1).
    # If A_list[i] = P, it means person (i+1) is behind person P.
    # Therefore, person P has person (i+1) right behind them.
    for i in range(N):
        person_who_is_behind = i + 1  # This is the 1-indexed person number
        person_in_front_of_them = A_list[i]

        if person_in_front_of_them == -1:
            # This person is at the very front of the line
            front_person = person_who_is_behind
        else:
            # 'person_who_is_behind' is right behind 'person_in_front_of_them'
            next_person[person_in_front_of_them] = person_who_is_behind

    # Reconstruct the line from front to back
    line = []
    current_person = front_person

    # Traverse the line using the next_person map.
    # The loop continues as long as the current person has someone behind them
    # (i.e., they are a key in the next_person map).
    while current_person in next_person:
        line.append(current_person)
        current_person = next_person[current_person]
    
    # After the loop, current_person is the last person in the line,
    # as they don't have anyone behind them (not a key in next_person).
    # Add this last person to the line.
    line.append(current_person)

    # Print the result.
    # Use map(str, line) to convert all integers to strings, then join them with spaces.
    sys.stdout.write(' '.join(map(str, line)) + '
')

solve()