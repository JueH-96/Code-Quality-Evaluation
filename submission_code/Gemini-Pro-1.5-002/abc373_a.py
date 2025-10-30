# YOUR CODE HERE
count = 0
for i in range(1, 13):
    s = input()
    if len(s) == i:
        count += 1
print(count)