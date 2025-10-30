N = int(input())
names = []
ages = []
for _ in range(N):
    s, a = input().split()
    names.append(s)
    ages.append(int(a))

# Find the index of the youngest person
min_age = min(ages)
start_index = ages.index(min_age)

# Create the order starting from the youngest person
order = names[start_index:] + names[:start_index]

# Print each name on a separate line
for name in order:
    print(name)