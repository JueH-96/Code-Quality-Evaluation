# YOUR CODE HERE
S = input().strip()
N = len(S)
for length in range(N, 0, -1):
    for i in range(N - length + 1):
        substr = S[i:i+length]
        if substr == substr[::-1]:
            print(length)
            exit()