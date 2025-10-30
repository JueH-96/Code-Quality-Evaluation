# YOUR CODE HERE

import sys

S = sys.stdin.readline().strip()

first_pipe = S.find('|')
second_pipe = S.rfind('|')

result = S[:first_pipe] + S[second_pipe+1:]

print(result)