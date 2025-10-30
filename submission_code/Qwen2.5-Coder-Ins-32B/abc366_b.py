# YOUR CODE HERE
import sys

def vertical_writing(N, strings):
    max_length = max(len(s) for s in strings)
    result = [''] * max_length
    
    for i, s in enumerate(strings):
        for j, char in enumerate(s):
            result[j] += char
        for j in range(len(s), max_length):
            result[j] += '*'
    
    # Remove trailing '*' from each line
    for i in range(max_length):
        while result[i] and result[i][-1] == '*':
            result[i] = result[i][:-1]
    
    return result

def main():
    input = sys.stdin.read().splitlines()
    N = int(input[0])
    strings = input[1:]
    result = vertical_writing(N, strings)
    for line in result:
        print(line)

if __name__ == "__main__":
    main()