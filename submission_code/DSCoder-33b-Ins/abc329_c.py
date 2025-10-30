# YOUR CODE HERE

N = int(input())
S = input()

count = 0
i = 0

while i < N:
    j = i
    while j < N and S[j] == S[i]:
        j += 1
    count += (j - i) * (j - i + 1) // 2
    i = j

print(count)