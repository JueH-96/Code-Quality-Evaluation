# YOUR CODE HERE
import sys

# Optional: Increase recursion depth if a recursive solution were used. Not needed here.
# sys.setrecursionlimit(3 * 10**5 + 10) # Example setting for large N

def solve():
    """
    Reads the input describing the relative positions of N people in a line,
    determines the absolute order of people from front to back,
    and prints this order.

    The input A provides predecessor information:
    - A[i] = -1 means person i+1 is at the front.
    - A[i] = p means person i+1 is right behind person p.

    The algorithm works by first transforming the predecessor information
    into successor information (who is behind whom), finding the front person,
    and then traversing the line from front to back using the successor links.
    """
    
    # Read the number of people N
    n = int(sys.stdin.readline())
    
    # Read the space-separated integers representing predecessor information.
    # a[i] is the person in front of person i+1 (using 1-based person indexing),
    # or -1 if person i+1 is at the front.
    # Python list 'a' is 0-indexed, so a[i] corresponds to person i+1.
    a = list(map(int, sys.stdin.readline().split()))

    # successor[p] will store the person number immediately behind person p.
    # We use an array of size n + 1 to accommodate 1-based indexing for people (1 to n).
    # Index 0 is unused. Initialize all entries to 0.
    # A value of 0 signifies that either the successor is not yet known or the person is last in line.
    successor = [0] * (n + 1)
    
    # Variable to store the number of the person identified as being at the front of the line.
    front_person = -1 # Initialize with an invalid value.

    # Process the input array 'a' to build the successor relationships and find the front person.
    # Iterate through each element of 'a', corresponding to persons 1 to N.
    for i in range(n):
        # person_num is the person number (1-based) whose information is at index i.
        person_num = i + 1
        # predecessor is the person in front of person_num, read from a[i].
        predecessor = a[i]

        if predecessor == -1:
            # If predecessor is -1, person_num is the one at the front of the line.
            # The problem guarantees exactly one such person.
            front_person = person_num
        else:
            # Otherwise, person 'person_num' is directly behind person 'predecessor'.
            # We record this relationship in the successor array:
            # The successor of 'predecessor' is 'person_num'.
            successor[predecessor] = person_num

    # Now, reconstruct the line order starting from the identified front person.
    result = [] # This list will store the sequence of people in the line, from front to back.
    current_person = front_person # Start the traversal from the front person.

    # The problem statement guarantees that the input forms exactly one line of N people.
    # Therefore, we can simply iterate N times to collect all people in order.
    # Start the traversal only if a front person was found (always true for N >= 1 per constraints).
    if front_person != -1: 
       for _ in range(n):
           # Append the current person to the result list.
           result.append(current_person)
           
           # Move to the next person in the line using the successor array.
           # successor[current_person] gives the number of the person behind the current one.
           # For the last person in line, successor[last_person] will be 0 (our initialization value),
           # which correctly terminates the conceptual linked list traversal after N steps.
           current_person = successor[current_person]

    # Print the final ordered list of people.
    # The '*' operator unpacks the list elements into separate arguments for print.
    # print separates arguments by spaces by default.
    print(*result)

# Execute the solve function when the script is run.
solve()