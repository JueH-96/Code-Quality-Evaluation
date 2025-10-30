import sys
input = sys.stdin.read

def main():
    S = input().strip()
    frequency = {}
    
    # Count the frequency of each character
    for char in S:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    
    # Find the character with the highest frequency that is earliest in alphabetical order
    max_freq = -1
    result_char = ''
    
    for char in sorted(frequency.keys()):  # Sorting ensures we check in alphabetical order
        if frequency[char] > max_freq:
            max_freq = frequency[char]
            result_char = char
    
    print(result_char)

main()