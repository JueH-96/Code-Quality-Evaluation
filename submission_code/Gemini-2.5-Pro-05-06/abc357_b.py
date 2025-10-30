S = input()
uppercase_count = 0
lowercase_count = 0

for char in S:
    if char.isupper():
        uppercase_count += 1
    elif char.islower():
        lowercase_count += 1
    # The problem states S consists only of lowercase and uppercase English letters,
    # so no other character types need to be handled.

if uppercase_count > lowercase_count:
    # If uppercase letters are more numerous, convert the entire string to uppercase.
    # This correctly converts lowercase letters to uppercase and leaves existing uppercase letters as they are.
    result_S = S.upper()
else:
    # Otherwise (i.e., lowercase_count > uppercase_count, since total length is odd, counts cannot be equal),
    # convert the entire string to lowercase.
    # This correctly converts uppercase letters to lowercase and leaves existing lowercase letters as they are.
    result_S = S.lower()

print(result_S)