# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
S = input[1]

t_count = 0
a_count = 0

for i in range(N):
    if S[i] == 'T':
        t_count += 1
    else:
        a_count += 1
    
    if t_count > a_count:
        print('T')
        break
    elif a_count > t_count:
        print('A')
        break

if t_count == a_count:
    print('T')