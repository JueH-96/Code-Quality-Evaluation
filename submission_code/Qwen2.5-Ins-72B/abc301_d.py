# YOUR CODE HERE
import sys

def solve(S, N):
    n = len(S)
    max_val = -1
    for i in range(2 ** S.count('?')):
        binary = bin(i)[2:].zfill(S.count('?'))
        j = 0
        current = ''
        for char in S:
            if char == '?':
                current += binary[j]
                j += 1
            else:
                current += char
        value = int(current, 2)
        if value <= N:
            max_val = max(max_val, value)
    print(max_val)

S = input().strip()
N = int(input().strip())
solve(S, N)