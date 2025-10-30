import sys

def solve():
    N = int(sys.stdin.readline())

    people = []
    for _ in range(N):
        # Read name (S) and age (A)
        s, a = sys.stdin.readline().split()
        people.append((s, int(a))) # Store as (name, age) tuples

    # Find the youngest person and their original index (0-based)
    min_age = float('inf')
    youngest_person_idx = -1

    for i in range(N):
        name, age = people[i]
        if age < min_age:
            min_age = age
            youngest_person_idx = i # Store the index of the youngest person

    # Print names in clockwise order starting from the youngest
    # The original input order (index 0 to N-1) represents clockwise seating.
    # We start from youngest_person_idx and wrap around using the modulo operator.
    for j in range(N):
        # Calculate the current index in the 'people' list
        # (youngest_person_idx + j) gives the logical position relative to the youngest.
        # % N makes it wrap around for circular order.
        current_print_idx = (youngest_person_idx + j) % N
        print(people[current_print_idx][0]) # Print only the name

solve()