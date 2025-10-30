import sys

def main():
    """
    Reads input: number of people N, followed by N lines of name and age.
    Finds the person with the minimum age.
    Prints the names of all people in clockwise order, starting from the youngest person.
    """
    # Read the number of people
    n = int(sys.stdin.readline())
    
    # Store person data (name, age) in a list. The index in the list corresponds 
    # to the initial seating position (0-based index for people 1 to N).
    persons = [] 
    for _ in range(n): # Loop N times to read each person's data
        # Read name (s) and age (as string a_str)
        s, a_str = sys.stdin.readline().split()
        # Convert age string to integer
        a = int(a_str)
        # Store the name and age as a dictionary in the list
        # Using a dictionary makes the code more readable (persons[i]['name'], persons[i]['age'])
        persons.append({'name': s, 'age': a}) 

    # Find the index (0 to N-1) corresponding to the youngest person.
    # Initialize min_age with a very large value (or the age of the first person)
    min_age = float('inf') 
    # Initialize youngest_start_index to an invalid value or 0 if initializing min_age with persons[0]['age']
    youngest_start_index = -1 

    # Iterate through the list of persons to find the minimum age and the index of that person
    for i in range(n):
        # If the current person's age is less than the minimum age found so far
        if persons[i]['age'] < min_age:
            # Update the minimum age
            min_age = persons[i]['age']
            # Update the index of the youngest person found so far
            youngest_start_index = i
            
    # Print the names in clockwise order starting from the youngest person.
    # We need to print N names in total.
    for i in range(n):
        # Calculate the index of the person to print in this step.
        # We start at youngest_start_index and move 'i' steps clockwise.
        # The modulo operator (%) ensures that the index wraps around correctly 
        # for the circular table (e.g., if N=5 and youngest_start_index=3, the indices printed are 3, 4, 0, 1, 2).
        current_person_index = (youngest_start_index + i) % n
        
        # Print the name of the person at the calculated index.
        # Each name is printed on a new line, as required by the output format.
        print(persons[current_person_index]['name'])

# Standard boilerplate to call the main function when the script is executed.
if __name__ == '__main__':
    main()