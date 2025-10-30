def min_dishes_to_eat(N, X, Y, sweetness, saltiness):
    # Sort dishes by sweetness in descending order
    sorted_by_sweetness = sorted(sweetness, reverse=True)
    
    # Calculate how many dishes are needed to exceed the sweetness threshold X
    total_sweetness = 0
    dishes_for_sweetness = N
    for i, s in enumerate(sorted_by_sweetness):
        total_sweetness += s
        if total_sweetness > X:
            dishes_for_sweetness = i + 1
            break
    
    # Sort dishes by saltiness in descending order
    sorted_by_saltiness = sorted(saltiness, reverse=True)
    
    # Calculate how many dishes are needed to exceed the saltiness threshold Y
    total_saltiness = 0
    dishes_for_saltiness = N
    for i, s in enumerate(sorted_by_saltiness):
        total_saltiness += s
        if total_saltiness > Y:
            dishes_for_saltiness = i + 1
            break
    
    # Return the minimum number of dishes needed
    return min(dishes_for_sweetness, dishes_for_saltiness)

# Read input
N, X, Y = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Solve and print output
result = min_dishes_to_eat(N, X, Y, A, B)
print(result)