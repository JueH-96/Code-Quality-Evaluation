# YOUR CODE HERE
n = int(input())
passengers = list(map(int, input().split()))

min_initial = 0
current = 0

for change in passengers:
    current += change
    if current < 0:
        min_initial += abs(current)
        current = 0

print(min_initial + current)