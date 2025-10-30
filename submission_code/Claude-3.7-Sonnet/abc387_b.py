def solve_multiplication_table(X):
    # Calculate total sum of all numbers in the 9x9 multiplication table
    total_sum = 0
    for i in range(1, 10):
        for j in range(1, 10):
            total_sum += i * j
    
    # Count occurrences of X in the table
    occurrences = 0
    for i in range(1, 10):
        for j in range(1, 10):
            if i * j == X:
                occurrences += 1
    
    # Subtract X × occurrences from the total sum
    result = total_sum - (X * occurrences)
    return result

# Read input
X = int(input())
print(solve_multiplication_table(X))