# YOUR CODE HERE
S = input().strip()
T = input().strip()

positions = []
s_idx = 0
t_idx = 0

while s_idx < len(S) and t_idx < len(T):
    if S[s_idx] == T[t_idx]:
        positions.append(t_idx + 1)  # 1-indexed position
        s_idx += 1
    t_idx += 1

print(' '.join(map(str, positions)))