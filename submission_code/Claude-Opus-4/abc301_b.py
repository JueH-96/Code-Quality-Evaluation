# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

while True:
    # Check if we need to continue
    need_insertion = False
    insert_pos = -1
    
    for i in range(len(a) - 1):
        if abs(a[i] - a[i+1]) != 1:
            need_insertion = True
            insert_pos = i
            break
    
    if not need_insertion:
        break
    
    # Perform insertion
    i = insert_pos
    if a[i] < a[i+1]:
        # Insert ascending sequence
        to_insert = list(range(a[i] + 1, a[i+1]))
    else:
        # Insert descending sequence
        to_insert = list(range(a[i] - 1, a[i+1], -1))
    
    # Insert the numbers
    a = a[:i+1] + to_insert + a[i+1:]

print(' '.join(map(str, a)))