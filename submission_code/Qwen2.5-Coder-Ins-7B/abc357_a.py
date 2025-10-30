# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
H = list(map(int, data[2:]))

count = 0
remaining_disinfectant = M

for hands in H:
    if hands <= remaining_disinfectant:
        count += 1
        remaining_disinfectant -= hands

print(count)