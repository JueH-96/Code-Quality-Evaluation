# YOUR CODE HERE
def can_win(s, t):
    s_count = {}
    t_count = {}
    at_count_s = 0
    at_count_t = 0

    for char in s:
        if char == '@':
            at_count_s += 1
        else:
            s_count[char] = s_count.get(char, 0) + 1

    for char in t:
        if char == '@':
            at_count_t += 1
        else:
            t_count[char] = t_count.get(char, 0) + 1

    atcoder = 'atcoder'
    diff = 0

    for char in set(s_count.keys()) | set(t_count.keys()):
        s_val = s_count.get(char, 0)
        t_val = t_count.get(char, 0)
        if s_val > t_val:
            diff += s_val - t_val
        elif t_val > s_val:
            diff += t_val - s_val

    at_available = min(at_count_s, at_count_t)
    at_needed = max(0, diff - at_available)

    return at_count_s + at_count_t >= at_needed

s = input().strip()
t = input().strip()

print("Yes" if can_win(s, t) else "No")