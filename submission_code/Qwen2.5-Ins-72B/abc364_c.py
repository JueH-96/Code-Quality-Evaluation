# YOUR CODE HERE
import sys

def min_dishes(N, X, Y, A, B):
    # Combine sweetness and saltiness into a list of tuples
    dishes = list(zip(A, B))
    
    # Sort dishes by the sum of sweetness and saltiness in descending order
    dishes.sort(key=lambda x: -(x[0] + x[1]))
    
    total_sweetness = 0
    total_saltiness = 0
    count = 0
    
    # Iterate through the sorted dishes
    for sweetness, saltiness in dishes:
        # Check if adding the current dish exceeds the limits
        if total_sweetness + sweetness > X or total_saltiness + saltiness > Y:
            break
        # Add the current dish to the total
        total_sweetness += sweetness
        total_saltiness += saltiness
        count += 1
    
    return count

# Read input
input = sys.stdin.read
N, X, Y, *AB = map(int, input().split())
A = AB[:N]
B = AB[N:]

# Solve and print the result
print(min_dishes(N, X, Y, A, B))