# YOUR CODE HERE
s = input().strip()
left = s.find('|')
right = s.find('|', left + 1)
print(s[:left] + s[right+1:])