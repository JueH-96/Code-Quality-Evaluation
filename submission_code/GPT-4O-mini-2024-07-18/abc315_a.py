# YOUR CODE HERE
def remove_vowels(s):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    result = ''.join(char for char in s if char not in vowels)
    return result

if __name__ == "__main__":
    import sys
    input_string = sys.stdin.read().strip()
    print(remove_vowels(input_string))