# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))
last_occurrence = {}
B = []
for idx, val in enumerate(A):
    i = idx + 1  # positions start from 1
    if val in last_occurrence:
        B.append(str(last_occurrence[val]))
    else:
        B.append(str(-1))
    last_occurrence[val] = i  # update last occurrence
print(' '.join(B))