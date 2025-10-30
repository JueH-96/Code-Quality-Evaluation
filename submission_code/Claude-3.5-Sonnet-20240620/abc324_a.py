# YOUR CODE HERE
N = int(input())
numbers = list(map(int, input().split()))

if len(set(numbers)) == 1:
    print("Yes")
else:
    print("No")