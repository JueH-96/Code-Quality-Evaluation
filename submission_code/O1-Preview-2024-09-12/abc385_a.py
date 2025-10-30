# YOUR CODE HERE
A, B, C = map(int, input().split())
result = "No"
if A + B == C or A + C == B or B + C == A:
    result = "Yes"
elif A == B == C:
    result = "Yes"
print(result)