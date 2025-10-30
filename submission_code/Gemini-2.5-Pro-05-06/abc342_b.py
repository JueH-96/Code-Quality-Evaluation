def solve():
    N = int(input())
    # P_list[i] stores the person number at the (i+1)-th position from the front.
    # For example, P_list[0] is person at 1st position, P_list[1] at 2nd, etc.
    P_list = list(map(int, input().split()))
    
    # Create a mapping from person number to their 1-indexed position in the line.
    # person_to_pos[person_id] = position
    # We use an array of size N+1 for 1-based indexing of person numbers (1 to N).
    person_to_pos = [0] * (N + 1) 
    for i in range(N): # i iterates from 0 to N-1
        person_num = P_list[i]
        position = i + 1 # The person P_list[i] is at (i+1)-th position
        person_to_pos[person_num] = position
        
    Q = int(input())
    
    # Process each query
    for _ in range(Q):
        # A and B are person numbers
        A, B = map(int, input().split())
        
        # Get their respective positions
        pos_A = person_to_pos[A]
        pos_B = person_to_pos[B]
        
        # Compare positions: smaller position index means further to the front.
        if pos_A < pos_B:
            print(A)  # Person A is further to the front
        else:
            # Since person A and B are distinct, their positions must be distinct.
            # So, if pos_A is not less than pos_B, then pos_B must be less than pos_A.
            print(B)  # Person B is further to the front

# Call the main function to execute the logic
solve()