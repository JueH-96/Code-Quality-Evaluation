# YOUR CODE HERE
M = int(input())
S1 = input()
S2 = input()
S3 = input()

# Check if all characters in any string are the same
def all_same(s):
    return all(char == s[0] for char in s)

# Check if it's possible to make all reels show the same character
if all_same(S1) and all_same(S2) and all_same(S3):
    if S1[0] == S2[0] == S3[0]:
        print(0)
    else:
        print(-1)
else:
    # Find the minimum time to make all reels show the same character
    for t in range(M):
        if S1[t] == S2[t] == S3[t]:
            print(t)
            break
    else:
        print(-1)