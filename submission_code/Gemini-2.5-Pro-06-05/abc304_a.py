import sys

def solve():
    """
    Solves the round table problem by finding the youngest person and printing
    the names in clockwise order starting from them.
    """
    try:
        # Read the number of people from standard input.
        n_str = sys.stdin.readline()
        if not n_str:
            return
        n = int(n_str)
    except (ValueError, IndexError):
        # This handles cases of malformed input for N, though constraints
        # guarantee valid input.
        return

    # Read the data for all N people into a list.
    # The list 'people' will store (name, age) tuples.
    # The index of the list corresponds to the person's original position minus 1.
    people = []
    for _ in range(n):
        line = sys.stdin.readline()
        if not line:
            continue
        name, age_str = line.split()
        people.append((name, int(age_str)))

    # Constraints guarantee N >= 2, so the list is not empty.
    # Find the index of the youngest person.
    # We initialize with the first person's data.
    min_age = people[0][1]
    start_index = 0
    
    # Iterate through the rest of the people to find the minimum age.
    for i in range(1, n):
        if people[i][1] < min_age:
            min_age = people[i][1]
            start_index = i

    # Print the names in clockwise order, starting from the youngest person.
    for i in range(n):
        # The modulo operator (%) handles the wrap-around for the circular table.
        # It calculates the correct index in the 'people' list for each step
        # in the clockwise traversal.
        current_index = (start_index + i) % n
        print(people[current_index][0])

# Execute the solution
solve()