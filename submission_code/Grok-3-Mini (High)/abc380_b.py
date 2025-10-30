# YOUR CODE HERE
import sys
S = sys.stdin.read().strip()
parts = S.split('|')
dash_parts = parts[1:-1]
A_lengths = [len(part) for part in dash_parts]
print(' '.join(map(str, A_lengths)))