# YOUR CODE HERE
N = int(input())
S = input()
count = 0
found = {'A': False, 'B': False, 'C': False}

for char in S:
    count += 1
    if char in found:
        found[char] = True
    if all(found.values()):
        break

print(count)