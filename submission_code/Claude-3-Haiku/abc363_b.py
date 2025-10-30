# Read the input
n, t, p = map(int, input().split())
hair_lengths = list(map(int, input().split()))

# Initialize the number of people with hair length >= t
count = sum(1 for length in hair_lengths if length >= t)

# Check if the condition is already satisfied
if count >= p:
    print(0)
    return

# Simulate the hair growth
days = 0
while True:
    days += 1
    for i in range(n):
        hair_lengths[i] += 1
    count = sum(1 for length in hair_lengths if length >= t)
    if count >= p:
        print(days)
        return