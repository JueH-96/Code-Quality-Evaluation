# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Check if all elements are equal by counting unique values
if len(set(a)) == 1:
    print("Yes")
else:
    print("No")