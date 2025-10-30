import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S_str = data[index]
index += 1
S = list(S_str)
# Set initial case state
if all(c.islower() for c in S):
    case_state = 1  # all lowercase
elif all(c.isupper() for c in S):
    case_state = 2  # all uppercase
else:
    case_state = 0  # mixed
Q = int(data[index])
index += 1
for _ in range(Q):
    t = int(data[index])
    index += 1
    x = int(data[index])  # x_i, may not be used
    index += 1
    c = data[index]  # character
    index += 1
    if t == 1:
        pos = x - 1  # 0-based index
        S[pos] = c
        if case_state == 1 and not c.islower():
            case_state = 0
        elif case_state == 2 and not c.isupper():
            case_state = 0
    elif t == 2:
        if case_state != 1:
            for i in range(N):
                if S[i].isupper():
                    S[i] = S[i].lower()
            case_state = 1
    elif t == 3:
        if case_state != 2:
            for i in range(N):
                if S[i].islower():
                    S[i] = S[i].upper()
            case_state = 2
# After all operations, join and print
result = ''.join(S)
print(result)