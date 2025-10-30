# YOUR CODE HERE
S = input().strip()
vowels = set('aeiou')
result = ''.join(char for char in S if char not in vowels)
print(result)