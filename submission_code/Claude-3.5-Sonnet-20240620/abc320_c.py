# YOUR CODE HERE
def find_min_time(M, S1, S2, S3):
    for t in range(M):
        char = S1[t % M]
        if (char in S2) and (char in S3):
            t2 = S2.index(char)
            t3 = S3.index(char)
            return max(t, t2, t3)
    return -1

M = int(input())
S1 = input()
S2 = input()
S3 = input()

result = find_min_time(M, S1, S2, S3)
print(result)