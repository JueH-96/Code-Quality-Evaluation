S = input().strip()

position = 1
insertions = 0

for char in S:
    if (position % 2 == 1 and char == 'i') or (position % 2 == 0 and char == 'o'):
        position += 1
    else:
        insertions += 1
        position += 2

if position % 2 == 0:
    insertions += 1

print(insertions)