# YOUR CODE HERE
N = int(input())
S = input()

total = 0
for i in range(N):
    current_value = 0
    for j in range(i, N):
        current_value = current_value * 10 + int(S[j])
        total += current_value

print(total)