# YOUR CODE HERE
n, t, p = map(int, input().split())
hair_lengths = list(map(int, input().split()))
days = 0
count = sum(1 for length in hair_lengths if length >= t)

while count < p:
    days += 1
    count = sum(1 for length in hair_lengths if length + days >= t)

print(days)