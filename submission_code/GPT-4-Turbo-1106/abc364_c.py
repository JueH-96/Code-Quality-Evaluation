def find_min_dishes(N, X, Y, sweetness, saltiness):
    # Initialize the total sweetness and saltiness
    total_sweetness = 0
    total_saltiness = 0
    # Initialize the number of dishes eaten
    dishes_eaten = 0
    
    # Sort the dishes by the ratio of sweetness to saltiness, in descending order
    dishes = sorted(zip(sweetness, saltiness), key=lambda x: -x[0]/x[1])
    
    # Iterate over the sorted dishes
    for sweet, salt in dishes:
        # Check if adding the current dish will exceed the sweetness or saltiness limit
        if total_sweetness + sweet > X or total_saltiness + salt > Y:
            break
        # Add the sweetness and saltiness of the current dish to the totals
        total_sweetness += sweet
        total_saltiness += salt
        # Increment the number of dishes eaten
        dishes_eaten += 1
    
    return dishes_eaten

# Read input from stdin
N, X, Y = map(int, input().split())
sweetness = list(map(int, input().split()))
saltiness = list(map(int, input().split()))

# Find the minimum possible number of dishes that will be eaten
result = find_min_dishes(N, X, Y, sweetness, saltiness)

# Write the answer to stdout
print(result)