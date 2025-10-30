# YOUR CODE HERE
n = int(input())
keys = [0, 0]
for i in range(n):
    a, s = input().split()
    a = int(a)
    if s == 'L':
        keys[0] = a
    else:
        keys[1] = a
    if len(set(keys)) == 1:
        continue
    move = min(abs(keys[0] - a), abs(keys[1] - a))
    keys[keys.index(min(keys))] += move
print(abs(keys[0] - 50) + abs(keys[1] - 50))