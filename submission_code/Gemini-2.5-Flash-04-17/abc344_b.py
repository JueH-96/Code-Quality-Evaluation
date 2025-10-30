# Read integers until 0 is encountered
numbers = []
while True:
    try:
        line = input()
        num = int(line)
        numbers.append(num)
        if num == 0:
            break
    except EOFError:
        # Should not happen based on problem description (guaranteed A_N = 0),
        # but good practice for input reading loops.
        break
    except ValueError:
        # Input is guaranteed to be integers, so this shouldn't happen.
        pass # Or handle error if necessary

# Print numbers in reverse order
# The list 'numbers' now contains [A_1, A_2, ..., A_N-1, A_N]
# We need to print A_N, A_N-1, ..., A_1
# Iterating through the list in reverse order and printing each element.
for num in reversed(numbers):
    print(num)