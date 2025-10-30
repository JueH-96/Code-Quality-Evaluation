from collections import defaultdict

def solve(S, X, Y):
    # Create a dictionary to store the cumulative sum of S
    S_sum = defaultdict(int)
    curr_sum = 0
    for c in S:
        curr_sum += ord(c) - ord('a')
        S_sum[curr_sum] += 1

    # Check if there exists a T that satisfies the condition
    curr_sum_x = 0
    curr_sum_y = 0
    for i in range(len(X)):
        if X[i] == '0':
            curr_sum_x += ord(S[0]) - ord('a')
        else:
            curr_sum_x -= ord(S[0]) - ord('a')

        if Y[i] == '0':
            curr_sum_y += ord(S[0]) - ord('a')
        else:
            curr_sum_y -= ord(S[0]) - ord('a')

        if curr_sum_x == curr_sum_y and S_sum[curr_sum_x] > 0:
            return "Yes"

    return "No"

# Read input and solve the problem
t = int(input())
for _ in range(t):
    S = input()
    X = input()
    Y = input()
    print(solve(S, X, Y))