import sys

input = sys.stdin.read
data = input().strip()

if data.endswith('san'):
    print('Yes')
else:
    print('No')