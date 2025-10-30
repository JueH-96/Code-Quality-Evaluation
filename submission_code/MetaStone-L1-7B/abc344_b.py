import sys

lines = []
for line in sys.stdin:
    line = line.strip()
    lines.append(int(line))
lines.reverse()
for num in lines:
    print(num)