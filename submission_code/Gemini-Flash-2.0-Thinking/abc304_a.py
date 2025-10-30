N = int(input())
names = []
ages = []
for _ in range(N):
    s, a = input().split()
    names.append(s)
    ages.append(int(a))

min_age = float('inf')
youngest_index = -1
for i in range(N):
    if ages[i] < min_age:
        min_age = ages[i]
        youngest_index = i

for i in range(N):
    current_index = (youngest_index + i) % N
    print(names[current_index])