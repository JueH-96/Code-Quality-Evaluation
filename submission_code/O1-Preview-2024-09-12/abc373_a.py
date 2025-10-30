# YOUR CODE HERE
count = 0
for i in range(1, 13):
    S_i = input()
    if len(S_i) == i:
        count += 1
print(count)