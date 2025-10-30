# YOUR CODE HERE
import sys

def can_win(S, T):
    allowed_chars = set('atcoder')
    s_count = {char: 0 for char in allowed_chars}
    t_count = {char: 0 for char in allowed_chars}
    s_at = 0
    t_at = 0

    for s, t in zip(S, T):
        if s == '@':
            s_at += 1
        elif s in allowed_chars:
            s_count[s] += 1
        if t == '@':
            t_at += 1
        elif t in allowed_chars:
            t_count[t] += 1

    for char in allowed_chars:
        if s_count[char] + s_at < t_count[char] or t_count[char] + t_at < s_count[char]:
            return False
        s_at -= max(0, t_count[char] - s_count[char])
        t_at -= max(0, s_count[char] - t_count[char])

    return s_at >= 0 and t_at >= 0

S = input().strip()
T = input().strip()

if can_win(S, T):
    print("Yes")
else:
    print("No")