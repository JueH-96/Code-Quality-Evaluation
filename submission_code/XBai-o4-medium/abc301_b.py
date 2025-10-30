n = int(input())
seq = list(map(int, input().split()))

while True:
    found = False
    for i in range(len(seq) - 1):
        if abs(seq[i] - seq[i+1]) != 1:
            found = True
            break
    if not found:
        break
    a_val = seq[i]
    b_val = seq[i+1]
    new_elements = []
    if a_val < b_val:
        new_elements = list(range(a_val + 1, b_val))
    else:
        new_elements = list(range(a_val - 1, b_val, -1))
    seq = seq[:i+1] + new_elements + seq[i+1:]

print(' '.join(map(str, seq)))