# YOUR CODE HERE
S = input().strip()
vowels = 'aeiou'
result = ''.join([char for char in S if char not in vowels])
print(result)