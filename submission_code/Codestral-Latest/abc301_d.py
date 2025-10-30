# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
N = int(data[1])

def find_max_value(S, N):
    max_value = -1

    def backtrack(index, current_value):
        nonlocal max_value
        if index == len(S):
            if current_value <= N:
                max_value = max(max_value, current_value)
            return

        if S[index] == '?':
            backtrack(index + 1, current_value * 2)
            backtrack(index + 1, current_value * 2 + 1)
        else:
            backtrack(index + 1, current_value * 2 + int(S[index]))

    backtrack(0, 0)
    return max_value

result = find_max_value(S, N)
print(result)