n = int(input())
a = list(map(int, input().split()))

while True:
    found = False
    for i in range(len(a) - 1):
        current = a[i]
        next_val = a[i + 1]
        if abs(current - next_val) == 1:
            continue
        # Need to insert elements
        found = True
        if current < next_val:
            insert = list(range(current + 1, next_val))
        else:
            insert = list(range(current - 1, next_val, -1))
        # Update the list
        a = a[:i+1] + insert + a[i+1:]
        break  # Break to restart the check from the beginning
    if not found:
        break

print(' '.join(map(str, a)))