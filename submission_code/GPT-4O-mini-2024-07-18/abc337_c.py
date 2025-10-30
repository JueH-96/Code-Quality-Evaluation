def arrange_people(N, A):
    # Create a list to hold the final arrangement
    line = [0] * N
    
    # Find the index of the person who is at the front of the line
    front_index = A.index(-1)
    line[0] = front_index + 1  # Store the person number (1-based index)
    
    # Build the line based on the given arrangement
    for i in range(1, N):
        person = i + 1  # Current person number (1-based index)
        behind_person = A[i]  # The person this one is behind
        if behind_person != -1:
            line[i] = behind_person  # Store the person number directly
    
    # Now we need to resolve the order based on the relationships
    result = []
    visited = [False] * (N + 1)  # To keep track of visited persons
    for i in range(N):
        if not visited[line[i]]:
            current = line[i]
            while current != -1 and not visited[current]:
                result.append(current)
                visited[current] = True
                # Move to the next person in line
                next_person = A[current - 1]  # A is 0-indexed, current is 1-indexed
                current = next_person if next_person != -1 else -1
    
    return result

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

result = arrange_people(N, A)
print(" ".join(map(str, result)))