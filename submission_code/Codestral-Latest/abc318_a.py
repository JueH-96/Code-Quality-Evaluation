# YOUR CODE HERE
N, M, P = map(int, input().split())

# Calculate the number of full moons Takahashi can see
count = 0
current_day = M

while current_day <= N:
    count += 1
    current_day += P

print(count)