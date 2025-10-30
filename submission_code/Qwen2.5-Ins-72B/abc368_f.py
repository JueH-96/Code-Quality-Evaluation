# YOUR CODE HERE
import sys

def find_winner(N, A):
    def is_power_of_two(x):
        return (x & (x - 1)) == 0

    xor_sum = 0
    for a in A:
        if not is_power_of_two(a):
            xor_sum ^= 1

    if xor_sum == 0:
        return "Bruno"
    else:
        return "Anna"

input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(find_winner(N, A))