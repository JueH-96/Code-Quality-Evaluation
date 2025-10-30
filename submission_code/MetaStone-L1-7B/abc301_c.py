S = input().strip()
T = input().strip()

allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}

s_counts = {c: 0 for c in allowed}
s_at = 0
for c in S:
    if c in allowed:
        s_counts[c] += 1
    elif c == '@':
        s_at += 1

t_counts = {c: 0 for c in allowed}
t_at = 0
for c in T:
    if c in allowed:
        t_counts[c] += 1
    elif c == '@':
        t_at += 1

# Check for non-allowed and non-@ characters
for c in set(S) | set(T):
    if c not in allowed and c != '@':
        if S.count(c) != T.count(c):
            print("No")
            exit()

required_s_replaced = 0
required_t_replaced = 0

for c in allowed:
    diff = t_counts[c] - s_counts[c]
    required_s_replaced += max(0, diff)
    required_t_replaced += max(0, -diff)

if required_s_replaced <= s_at and required_t_replaced <= t_at:
    print("Yes")
else:
    print("No")