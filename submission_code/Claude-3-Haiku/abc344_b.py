# Read the inputs from stdin
inputs = []
while True:
    try:
        a = int(input())
        inputs.append(a)
    except EOFError:
        break

# Reverse the list and print the elements
for i in range(len(inputs)-1, -1, -1):
    print(inputs[i])