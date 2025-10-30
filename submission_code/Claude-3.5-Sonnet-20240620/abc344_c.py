# YOUR CODE HERE
def read_sequence():
    return set(map(int, input().split()))

def solve(A, B, C, X):
    for a in A:
        for b in B:
            if X - a - b in C:
                return "Yes"
    return "No"

N = int(input())
A = read_sequence()
M = int(input())
B = read_sequence()
L = int(input())
C = read_sequence()
Q = int(input())
X = list(map(int, input().split()))

for x in X:
    print(solve(A, B, C, x))