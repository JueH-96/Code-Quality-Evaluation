# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

min_passengers = min(A)

if min_passengers < 0:
    min_passengers = abs(min_passengers)
else:
    min_passengers = 0

print(min_passengers)