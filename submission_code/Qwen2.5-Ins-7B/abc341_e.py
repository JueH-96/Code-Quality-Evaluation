# YOUR CODE HERE
from sys import stdin, stdout

N, Q = map(int, stdin.readline().split())
S = list(map(int, stdin.readline().strip()))
queries = []
for _ in range(Q):
    queries.append(list(map(int, stdin.readline().split())))

def flip_range(start, end):
    for i in range(start, end + 1):
        S[i] = 1 - S[i]

def is_good_string(start, end):
    for i in range(start, end):
        if S[i] == S[i + 1]:
            return False
    return True

for query in queries:
    if query[0] == 1:
        flip_range(query[1] - 1, query[2] - 1)
    else:
        if is_good_string(query[1] - 1, query[2] - 1):
            stdout.write("Yes
")
        else:
            stdout.write("No
")