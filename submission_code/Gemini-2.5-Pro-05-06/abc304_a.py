import sys

def main():
    # Read N, the number of people
    N = int(sys.stdin.readline())

    # Store people's data. Each element is a tuple (name, age).
    # The list index corresponds to a 0-based version of the person's input order.
    # This order also represents their clockwise seating arrangement.
    people = []
    for _ in range(N):
        name, age_str = sys.stdin.readline().split()
        age = int(age_str)
        people.append((name, age))

    # Find the 0-based index of the youngest person in the 'people' list.
    # N is guaranteed to be at least 2 by constraints, so 'people' is not empty.
    # The 'key' argument to min specifies a function to be called on each element
    # of 'range(N)' (which are indices 0, 1, ..., N-1).
    # We want the index 'idx' for which 'people[idx][1]' (the age) is minimal.
    youngest_person_start_idx = min(range(N), key=lambda idx: people[idx][1])

    # Print N names in clockwise order, starting from the youngest.
    # The loop runs N times, once for each person to be printed.
    # 'i' represents the offset from the youngest person (0 for the youngest, 
    # 1 for the person clockwise to the youngest, and so on).
    for i in range(N):
        # Calculate the actual index in the 'people' list for the current person.
        # This involves adding the offset 'i' to 'youngest_person_start_idx' and
        # wrapping around using the modulo N operator for the circular table.
        current_person_list_idx = (youngest_person_start_idx + i) % N
        
        # Print the name of the person at the calculated index.
        # The name is the first element (index 0) of the tuple (name, age).
        print(people[current_person_list_idx][0])

if __name__ == '__main__':
    main()