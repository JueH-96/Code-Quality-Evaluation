# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]
queries = [(int(data[2 + 2 * i]) - 1, data[3 + 2 * i]) for i in range(Q)]

def count_abc(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i:i+3] == "ABC":
            count += 1
    return count

result = []
current_count = count_abc(S)
result.append(current_count)

for x, c in queries:
    if S[x] == c:
        result.append(current_count)
    else:
        if S[x:x+3] == "ABC":
            current_count -= 1
        if S[x-1:x+2] == "ABC":
            current_count += 1
        if S[x:x+3] == "ABC":
            current_count += 1
        if S[x-1:x+2] == "ABC":
            current_count -= 1
        S = S[:x] + c + S[x+1:]
        result.append(current_count)

for r in result:
    print(r)