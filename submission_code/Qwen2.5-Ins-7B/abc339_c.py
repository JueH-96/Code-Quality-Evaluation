# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
current_passengers = 0
min_passengers = 0
for i in range(n):
    current_passengers += a[i]
    min_passengers = min(min_passengers, current_passengers)
print(-min_passengers)