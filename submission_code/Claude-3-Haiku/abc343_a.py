# Read the input
A, B = map(int, input().split())

# Find a number between 0 and 9 that is not equal to A + B
for i in range(10):
    if i != A + B:
        print(i)
        break