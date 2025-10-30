# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
M = int(data[N+1])
B = list(map(int, data[N+2:N+M+2]))
L = int(data[N+M+2])
C = list(map(int, data[N+M+3:N+M+3+L]))
Q = int(data[N+M+3+L])
X = list(map(int, data[N+M+3+L+1:]))

# Create sets for quick lookup
set_A = set(A)
set_B = set(B)
set_C = set(C)

# Function to check if there exists a combination (a, b, c) such that a + b + c = x
def can_form_sum(x):
    for a in set_A:
        for b in set_B:
            if x - a - b in set_C:
                return "Yes"
    return "No"

# Process each query
results = [can_form_sum(x) for x in X]

# Print results
for result in results:
    print(result)