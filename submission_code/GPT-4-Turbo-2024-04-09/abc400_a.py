# YOUR CODE HERE
A = int(input().strip())
total_people = 400

if total_people % A == 0:
    B = total_people // A
    print(B)
else:
    print(-1)