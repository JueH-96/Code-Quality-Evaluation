s = input().strip()
pointer = 0
insertions = 0
current_target = 1  # current position being checked (starts at 1)

while pointer < len(s):
    expected = 'i' if (current_target % 2 == 1) else 'o'
    if s[pointer] == expected:
        pointer += 1
    else:
        insertions += 1
    current_target += 1

# Check if the total length is even
length = current_target - 1
if length % 2 != 0:
    insertions += 1

print(insertions)