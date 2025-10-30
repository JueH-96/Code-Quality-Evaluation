# YOUR CODE HERE

n, x, y = map(int, input().split())
dishes = [tuple(map(int, input().split())) for _ in range(n)]

# Sort the dishes by the sum of sweetness and saltiness
dishes.sort(key=lambda x: x[0] + x[1], reverse=True)

# Initialize the total sweetness and saltiness
total_sweetness = 0
total_saltiness = 0

# Initialize the count of dishes
count = 0

# Iterate over the sorted dishes
for dish in dishes:
    # If the total sweetness and saltiness does not exceed the limit
    if total_sweetness + dish[0] <= x and total_saltiness + dish[1] <= y:
        # Add the sweetness and saltiness to the total
        total_sweetness += dish[0]
        total_saltiness += dish[1]
        # Increment the count of dishes
        count += 1

# Print the count of dishes
print(count)