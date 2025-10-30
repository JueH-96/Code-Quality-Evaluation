# YOUR CODE HERE

t = int(input())

for _ in range(t):
    s = input()
    if s == "abc" or s == "acb" or s == "bac" or s == "cab":
        print("YES")
    else:
        print("NO")