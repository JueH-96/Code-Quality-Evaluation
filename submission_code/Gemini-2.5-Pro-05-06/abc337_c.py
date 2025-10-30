import sys

def main():
    N = int(sys.stdin.readline())
    # A_input is a 0-indexed list where A_input[i] corresponds to
    # the value A_{i+1} in the problem statement.
    # This value A_{i+1} is the information for person i+1.
    A_input = list(map(int, sys.stdin.readline().split()))

    # next_node[p] will store the ID of the person immediately behind person p.
    # Person IDs are 1-indexed (from 1 to N).
    # The array is of size N+1, and index 0 is unused.
    # Initialize with 0, signifying no known person is behind that person yet.
    # If a person is last in line, their entry in next_node will remain 0.
    next_node = [0] * (N + 1) 
    
    # head_of_line will store the ID of the person at the front of the line.
    head_of_line = -1 # Default/uninitialized state

    # Iterate through the input information.
    # The loop variable 'i' goes from 0 to N-1.
    # 'person_id' will be i+1, so it goes from 1 to N.
    for i in range(N):
        person_id = i + 1  # Current person's ID (1-indexed)
        
        # person_in_front_of_id is the ID of the person in front of person_id.
        # This value is A_input[i] (which is A_{person_id} from problem statement).
        person_in_front_of_id = A_input[i]

        if person_in_front_of_id == -1:
            # If A_{person_id} is -1, then person_id is at the front of the line.
            head_of_line = person_id
        else:
            # Otherwise, person_id is right behind person_in_front_of_id.
            # This means person_in_front_of_id is followed by person_id.
            # We record this in our next_node array.
            next_node[person_in_front_of_id] = person_id
            
    # Reconstruct the line starting from head_of_line.
    ordered_line = []
    current_person_in_line = head_of_line
    
    # The line consists of N people. We trace them one by one.
    for _ in range(N):
        ordered_line.append(current_person_in_line)
        # Move to the next person in the line using the next_node array.
        current_person_in_line = next_node[current_person_in_line]
        # If current_person_in_line was the last person, next_node[current_person_in_line]
        # would be 0. This 0 might be assigned to current_person_in_line in the last
        # iteration, but it won't be appended to ordered_line or used as an index.
        
    # Print the reconstructed line.
    # Elements of ordered_line are integers; they need to be converted to strings
    # and joined by spaces.
    sys.stdout.write(" ".join(map(str, ordered_line)) + "
")

if __name__ == '__main__':
    main()