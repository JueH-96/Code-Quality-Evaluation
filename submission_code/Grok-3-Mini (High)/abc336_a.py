# Read the input from standard input
N = int(input())

# Construct the Dragon String: 'L' followed by N 'o's, then 'n', then 'g'
dragon_string = 'L' + 'o' * N + 'n' + 'g'

# Print the Dragon String to standard output
print(dragon_string)