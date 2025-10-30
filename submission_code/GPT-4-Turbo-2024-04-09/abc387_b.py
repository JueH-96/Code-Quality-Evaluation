# YOUR CODE HERE
X = int(input())

total_sum = 0
count_X = 0

# Calculate the sum of all elements in the 9x9 multiplication table
# and count how many times X appears
for i in range(1, 10):
    for j in range(1, 10):
        product = i * j
        total_sum += product
        if product == X:
            count_X += 1

# Subtract the sum of all occurrences of X from the total sum
result = total_sum - count_X * X

# Output the result
print(result)