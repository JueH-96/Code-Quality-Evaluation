# Read the input
N, Q = map(int, input().split())
S = input()

# Function to solve the problem for a given query
def solve_query(l, r):
    count = 0
    for i in range(l-1, r-1):
        if S[i] == S[i+1]:
            count += 1
    return count

# Solve each query and print the answer
for _ in range(Q):
    l, r = map(int, input().split())
    print(solve_query(l, r))