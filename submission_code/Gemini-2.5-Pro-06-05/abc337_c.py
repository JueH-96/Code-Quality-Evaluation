import sys

def solve():
    """
    This function reads the input, solves the problem, and prints the output.
    """
    # Read N from standard input.
    # Using sys.stdin.readline is generally faster for large inputs in competitive programming.
    try:
        N = int(sys.stdin.readline())
        # Read the sequence A from standard input.
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle cases with empty or malformed input.
        return

    # The problem can be modeled as a singly linked list.
    # The input A[i] (for person i+1) gives the person in front.
    # This means `A[i]` is the 'predecessor' of `i+1`.
    # e.g., if A[0] = 4, person 1 is behind person 4. Relationship: 4 -> 1.

    # To traverse from front to back, we need 'successor' links.
    # If `pred` -> `person`, then the successor of `pred` is `person`.
    # We'll build a `successors` array to store this forward-pointing information.
    # `successors[p]` will store the ID of the person standing right behind person `p`.
    # Person IDs are 1-based (1 to N), so the array size is N+1.
    # We initialize it with a sentinel value (0), as person IDs are always >= 1.
    successors = [0] * (N + 1)

    # We also need to find the person at the very front of the line (the head).
    head = -1

    # Iterate through the input array A to populate our `successors` array and find the head.
    # The input list A is 0-indexed, so A[i] corresponds to person i+1.
    # We use enumerate to get both the index `i` and the value `pred`.
    for i, pred in enumerate(A):
        person = i + 1
        if pred == -1:
            # This person has no one in front, so they are the head of the line.
            head = person
        else:
            # Person `person` is behind person `pred`.
            # So, the successor of `pred` is `person`.
            successors[pred] = person

    # Now we have the head of the line and the successor of each person.
    # We can traverse the entire line starting from the head.
    line_order = []
    current_person = head

    # The line has N people, so we traverse N steps to visit everyone.
    for _ in range(N):
        line_order.append(current_person)
        # Move to the next person in the line.
        current_person = successors[current_person]

    # Print the final ordered list, with elements separated by spaces.
    # The '*' operator unpacks the list elements as arguments to print().
    print(*line_order)

solve()