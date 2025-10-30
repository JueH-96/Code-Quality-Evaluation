# Read all lines until an empty line is encountered
numbers = []
while True:
    try:
        line = input()
        if line.strip() == '':
            break
        numbers.append(int(line))
    except EOFError:
        break

# Print the numbers in reverse order
for num in reversed(numbers):
    print(num)