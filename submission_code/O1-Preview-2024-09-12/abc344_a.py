# YOUR CODE HERE
s = input()
first_index = s.find('|')
last_index = s.rfind('|')
result = s[:first_index] + s[last_index+1:]
print(result)