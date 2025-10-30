# YOUR CODE HERE
s = input()
first = s.find('|')
second = s.rfind('|')
print(s[:first] + s[second+1:])