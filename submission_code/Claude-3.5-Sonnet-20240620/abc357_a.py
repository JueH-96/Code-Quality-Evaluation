# YOUR CODE HERE
N, M = map(int, input().split())
H = list(map(int, input().split()))

aliens_disinfected = 0
remaining_disinfectant = M

for hands in H:
    if remaining_disinfectant >= hands:
        aliens_disinfected += 1
        remaining_disinfectant -= hands
    else:
        break

print(aliens_disinfected)