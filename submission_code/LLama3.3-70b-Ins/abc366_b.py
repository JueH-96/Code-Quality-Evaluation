import sys

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    n = int(lines[0].strip())
    strings = [line.strip() for line in lines[1:]]
    return n, strings

def convert_to_vertical(n, strings):
    """Convert horizontally written text to vertical writing."""
    max_length = max(len(s) for s in strings)
    result = [['*'] * n for _ in range(max_length)]

    for i, s in enumerate(strings):
        for j, c in enumerate(s):
            result[j][n - i - 1] = c

    # Remove trailing '*' from each string
    result = [''.join(s) for s in result]
    result = [s.rstrip('*') + s[len(s.rstrip('*')):] for s in result]

    return result

def main():
    n, strings = read_input()
    result = convert_to_vertical(n, strings)
    for s in result:
        print(s)

if __name__ == "__main__":
    main()