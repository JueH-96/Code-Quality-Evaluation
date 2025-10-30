# Read the input
n = int(input())

# Initialize the result list
result = []

# Iterate through each penalty kick from 1 to N
for i in range(1, n + 1):
    if i % 3 == 0:
        result.append('x')
    else:
        result.append('o')

# Join the list into a string and print
print(''.join(result))