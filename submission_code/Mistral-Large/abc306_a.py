import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

result = ''.join(char * 2 for char in S)

sys.stdout.write(result)