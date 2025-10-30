# YOUR CODE HERE
N = int(input())
S = input()
T = input()

hamming_distance = 0
for i in range(N):
    if S[i] != T[i]:
        hamming_distance += 1

print(hamming_distance)