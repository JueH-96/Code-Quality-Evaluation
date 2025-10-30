# Read the input
N = int(input())
Q = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]

# Calculate the maximum number of servings of each dish
max_a = min([q // a for q, a in zip(Q, A) if a > 0])
max_b = min([q // b for q, b in zip(Q, B) if b > 0])

# Calculate the total number of servings
total_servings = max_a + max_b

# Print the result
print(total_servings)