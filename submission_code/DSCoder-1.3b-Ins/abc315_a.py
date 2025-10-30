# YOUR CODE HERE
S = input()
new_S = [char for char in S if char not in 'aeiou']
print(''.join(new_S))