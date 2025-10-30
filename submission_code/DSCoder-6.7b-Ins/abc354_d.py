A, B, C, D = map(int, input().split())

# Calculate the number of black squares along the x-axis
x_black = (C - A) // 2

# Calculate the number of black squares along the y-axis
y_black = (D - B) // 2

# Calculate the total number of black squares
total_black = x_black * y_black

# Double the total number of black squares and print the result
print(total_black * 2)