from fractions import Fraction

n = int(input())
people = []

for i in range(n):
    a, b = map(int, input().split())
    success_rate = Fraction(a, a + b)
    people.append((success_rate, i + 1))

# Sort by success rate in descending order, and by person number in ascending order for ties
people.sort(key=lambda x: (-x[0], x[1]))

# Print the person numbers
result = [str(person[1]) for person in people]
print(' '.join(result))