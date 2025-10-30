# YOUR CODE HERE
def remove_vowels(S):
    vowels = set('aeiou')
    result = ''.join([char for char in S if char not in vowels])
    return result

if __name__ == "__main__":
    import sys
    input = sys.stdin.read().strip()
    print(remove_vowels(input))