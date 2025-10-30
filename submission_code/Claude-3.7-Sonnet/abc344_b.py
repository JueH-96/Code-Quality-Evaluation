# Read integers until we encounter 0
integers = []
while True:
    try:
        num = int(input())
        integers.append(num)
        if num == 0:
            break
    except EOFError:
        break

# Print the integers in reverse order
for num in reversed(integers):
    print(num)