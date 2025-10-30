import sys

lines = []
for line in sys.stdin:
    lines.append(int(line.strip()))
reversed_lines = lines[::-1]
for num in reversed_lines:
    print(num)