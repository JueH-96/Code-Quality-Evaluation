# YOUR CODE HERE
X = float(input())

if X >= 38.0:
    print(1)
elif X >= 37.5: # This implies 37.5 <= X < 38.0 because the first condition was false
    print(2)
else: # This implies X < 37.5
    print(3)