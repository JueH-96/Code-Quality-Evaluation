# YOUR CODE HERE

N = int(input())
binary = bin(N)[2:]
count = 0
for i in reversed(binary):
    if i == '0':
        count += 1
    else:
        break
print(count)