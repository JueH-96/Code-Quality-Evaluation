# YOUR CODE HERE
N, M = map(int, input().split())
H_list = list(map(int, input().split()))

count = 0
disinfectant = M

for H_i in H_list:
    if disinfectant == 0:
        break
    if disinfectant >= H_i:
        disinfectant -= H_i
        count +=1
    else:
        # Not enough disinfectant to disinfect all hands
        # They use up the remaining disinfectant
        disinfectant = 0
        # Do not increment the count, as they didn't disinfect all their hands

print(count)