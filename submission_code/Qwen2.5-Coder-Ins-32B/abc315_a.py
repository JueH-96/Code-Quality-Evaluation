# YOUR CODE HERE
import sys

def remove_vowels(s):
    vowels = set('aeiou')
    return ''.join(char for char in s if char not in vowels)

input_string = sys.stdin.read().strip()
result = remove_vowels(input_string)
print(result)