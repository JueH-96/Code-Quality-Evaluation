# YOUR CODE HERE
import sys
import math

N = int(sys.stdin.readline())

bits = N.bit_length()
S = N * bits - (2**bits - N)

print(S)