# Read the input
N = int(input())
S = input()
T = input()

# Calculate the Hamming distance
distance = 0
for i in range(N):
    if S[i] != T[i]:
        distance += 1

# Print the answer
print(distance)