# Read the input integer N
N = int(input())

# Construct the Dragon String
# The string starts with 'L'
dragon_string = 'L'
# Followed by N occurrences of 'o'
dragon_string += 'o' * N
# Followed by 'n'
dragon_string += 'n'
# Followed by 'g'
dragon_string += 'g'

# Print the resulting Dragon String
print(dragon_string)