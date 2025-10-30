import sys

def vertical_writing(strings):
    max_length = max(len(s) for s in strings)
    vertical_strings = ['' for _ in range(max_length)]
    
    for i, s in enumerate(strings):
        for j in range(max_length):
            if j < len(s):
                vertical_strings[j] += s[j]
            else:
                vertical_strings[j] += '*'
    
    # Remove trailing '*' from each vertical string
    vertical_strings = [s.rstrip('*') for s in vertical_strings]
    
    return vertical_strings

if __name__ == "__main__":
    input_lines = sys.stdin.read().splitlines()
    N = int(input_lines[0])
    strings = input_lines[1:N+1]
    
    result = vertical_writing(strings)
    for line in result:
        print(line)