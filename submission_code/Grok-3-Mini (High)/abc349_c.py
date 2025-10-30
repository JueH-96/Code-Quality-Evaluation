import sys

def is_subsequence(target, string):
    target_idx = 0
    for char in string:
        if target_idx < len(target) and char == target[target_idx]:
            target_idx += 1
    return target_idx == len(target)

S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()
t_lower = T.lower()

if is_subsequence(t_lower, S):
    print("Yes")
elif t_lower[2] == 'x' and is_subsequence(t_lower[:2], S):
    print("Yes")
else:
    print("No")