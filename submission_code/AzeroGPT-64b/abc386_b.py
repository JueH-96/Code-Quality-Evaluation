import re
S = input()
result = 0
# Split the input string by any sequence of zeros
parts = re.split('0+', S)
for i, part in enumerate(parts):
    # For the first part, add the length of the part as the number of button presses
    if i == 0:
        result += len(part)
    else:
        # For subsequent parts (after a sequence of zeros), add the number of zeros
        # from the previous sequence plus one for the non-zero digit
        result += len(parts[i-1])+1

print(result)