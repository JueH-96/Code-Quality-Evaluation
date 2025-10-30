import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
H = list(map(int, data[2:]))

remaining_disinfectant = M
aliens_disinfected = 0

for hands in H:
    if remaining_disinfectant >= hands:
        remaining_disinfectant -= hands
        aliens_disinfected += 1
    else:
        break

print(aliens_disinfected)