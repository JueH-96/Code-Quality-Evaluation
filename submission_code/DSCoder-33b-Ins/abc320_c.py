# YOUR CODE HERE

M = int(input())
S1 = input()
S2 = input()
S3 = input()

def solve(M, S1, S2, S3):
    for i in range(M):
        if S1[i] == S2[i] == S3[i]:
            return i
    return -1

print(solve(M, S1, S2, S3))