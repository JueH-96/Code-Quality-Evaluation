import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read the initial arrangement of people P
    # P[i] gives the ID of the person at position (i+1) from the front.
    # For example, P[0] is the person at the 1st position.
    P = list(map(int, sys.stdin.readline().split()))

    # Create a mapping from person ID to their position in the line.
    # This allows for O(1) lookup of a person's position given their ID.
    # Since person IDs are from 1 to N, we can use a list of size N+1.
    # person_to_pos[person_id] will store the 1-indexed position of that person.
    person_to_pos = [0] * (N + 1)
    
    # Populate the person_to_pos mapping
    for i in range(N):
        person_id = P[i]
        # The person P[i] is at the (i+1)-th position from the front.
        position = i + 1 
        person_to_pos[person_id] = position
        
    # Read Q, the number of queries
    Q = int(sys.stdin.readline())
    
    # List to store the results of each query
    results = []
    
    # Process each query
    for _ in range(Q):
        # Read the two person IDs for the current query
        A, B = map(int, sys.stdin.readline().split())
        
        # Get the positions of person A and person B from our precomputed map
        pos_A = person_to_pos[A]
        pos_B = person_to_pos[B]
        
        # Compare their positions to find who is further to the front
        if pos_A < pos_B:
            # If A's position is smaller, A is further to the front
            results.append(str(A))
        else:
            # Otherwise, B is further to the front
            results.append(str(B))
            
    # Print all accumulated results, each on a new line.
    # Using sys.stdout.write for potentially faster output.
    sys.stdout.write("
".join(results) + "
")

# Call the solve function to run the program
solve()