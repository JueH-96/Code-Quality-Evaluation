# YOUR CODE HERE
n = int(input())
people = []

for i in range(n):
    a, b = map(int, input().split())
    success_rate = a / (a + b)
    people.append((success_rate, i + 1))  # (success_rate, person_number)

# Sort by success rate descending, then by person number ascending
people.sort(key=lambda x: (-x[0], x[1]))

# Print the person numbers
result = [str(person[1]) for person in people]
print(' '.join(result))