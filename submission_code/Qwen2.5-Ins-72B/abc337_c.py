# YOUR CODE HERE
import sys

def find_order(N, A):
    # Find the person at the front of the line
    front_person = A.index(-1) + 1
    
    # Initialize the result list with the front person
    result = [front_person]
    
    # Start from the front person and follow the sequence
    current_person = front_person
    while True:
        next_person = A[current_person - 1]
        if next_person == -1:
            break
        result.append(next_person)
        current_person = next_person
    
    return result

# Read input
N = int(input())
A = list(map(int, input().split()))

# Find the order and print it
order = find_order(N, A)
print(' '.join(map(str, order)))