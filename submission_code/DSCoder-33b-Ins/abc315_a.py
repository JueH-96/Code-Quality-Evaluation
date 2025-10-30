# YOUR CODE HERE

S = input()
vowels = ['a', 'e', 'i', 'o', 'u']
result = ''

for char in S:
    if char not in vowels:
        result += char

print(result)