# YOUR CODE HERE
N = int(input())
S = list(input())
Q = int(input())

# Track the last case conversion operation
last_case_op = -1
last_case_type = 0

operations = []
for i in range(Q):
    parts = input().split()
    t = int(parts[0])
    x = int(parts[1])
    c = parts[2]
    operations.append((t, x, c, i))
    
    if t == 2 or t == 3:
        last_case_op = i
        last_case_type = t

# Apply operations
for i, (t, x, c, idx) in enumerate(operations):
    if t == 1:
        S[x-1] = c
        # If there's a case operation after this, we need to apply it to this character
        if idx < last_case_op:
            if last_case_type == 2:
                S[x-1] = S[x-1].lower()
            elif last_case_type == 3:
                S[x-1] = S[x-1].upper()

# Apply the final case conversion if any
if last_case_op != -1:
    if last_case_type == 2:
        S = [ch.lower() for ch in S]
    elif last_case_type == 3:
        S = [ch.upper() for ch in S]

print(''.join(S))