n = int(input())
sequence = list(map(int, input().split()))

while True:
    found = False
    insert_pos = -1
    for i in range(len(sequence) - 1):
        if abs(sequence[i] - sequence[i+1]) != 1:
            found = True
            insert_pos = i
            break
    if not found:
        break
    a = sequence[insert_pos]
    b = sequence[insert_pos + 1]
    if a < b:
        inserted = list(range(a + 1, b))
    else:
        inserted = list(range(a - 1, b, -1))
    sequence = sequence[:insert_pos+1] + inserted + sequence[insert_pos+1:]

print(' '.join(map(str, sequence)))