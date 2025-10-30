# YOUR CODE HERE
t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    if n % 3 == 0:
        print("First")
    else:
        print("Second")