# YOUR CODE HERE
N, X, Y = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create list of dishes with their indices
dishes = []
for i in range(N):
    # Calculate the fraction of limit each dish represents
    # We use max to prioritize dishes that hit either limit faster
    ratio = max(A[i]/X, B[i]/Y)
    dishes.append((ratio, A[i], B[i]))

# Sort by ratio in descending order
dishes.sort(reverse=True)

# Simulate eating dishes in this order
total_sweetness = 0
total_saltiness = 0
count = 0

for ratio, sweetness, saltiness in dishes:
    total_sweetness += sweetness
    total_saltiness += saltiness
    count += 1
    
    if total_sweetness > X or total_saltiness > Y:
        break

print(count)