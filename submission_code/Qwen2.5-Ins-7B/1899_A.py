# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n = int(input())
    if n % 3 == 0:
        print("First")
    elif n % 3 == 1:
        if (10 - (n - 1) % 2) >= 0:
            print("First")
        else:
            print("Second")
    else:
        if (10 - (n + 1) % 2) >= 0:
            print("First")
        else:
            print("Second")