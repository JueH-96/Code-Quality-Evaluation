# YOUR CODE HERE
S = input().strip()
vowels = {'a', 'e', 'i', 'o', 'u'}
result = ''.join([char for char in S if char not in vowels])
print(result)