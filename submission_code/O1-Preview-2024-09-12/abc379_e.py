# YOUR CODE HERE
N = int(input())
S = input().strip()
total = 0
last_value = 0

for p in range(N):
    digit = int(S[p])
    position = p + 1
    last_value = last_value * 10 + digit * position
    total += last_value

print(total)