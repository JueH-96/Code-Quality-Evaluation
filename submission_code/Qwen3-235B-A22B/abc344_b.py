import sys

data = [int(line.strip()) for line in sys.stdin]
for num in reversed(data):
    print(num)