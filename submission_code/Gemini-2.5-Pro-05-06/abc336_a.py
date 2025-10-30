# YOUR CODE HERE
N = int(input())
# Construct the Dragon String:
# It starts with 'L'
# followed by N occurrences of 'o'
# followed by 'n'
# followed by 'g'
dragon_string = 'L' + ('o' * N) + 'n' + 'g'
print(dragon_string)