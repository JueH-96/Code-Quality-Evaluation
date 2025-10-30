# Read the input rating R from standard input
R = int(input())

# Calculate the next threshold where the number of ^ increases
# The next threshold is the smallest multiple of 100 greater than R
next_thresh = ((R // 100) + 1) * 100

# Calculate the minimum increase needed
increase = next_thresh - R

# Print the result
print(increase)