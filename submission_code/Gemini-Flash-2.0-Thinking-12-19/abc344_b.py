# YOUR CODE HERE
inputs = []
while True:
    line = input()
    num = int(line)
    inputs.append(num)
    if num == 0:
        break

inputs.reverse()
for num in inputs:
    print(num)