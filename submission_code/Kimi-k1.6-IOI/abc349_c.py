def is_subsequence(target, s):
    j = 0
    m = len(target)
    for char in s:
        if j < m and char == target[j]:
            j += 1
    return j == m

s = input().strip()
t = input().strip()

t_lower = t.lower()

# Check for 3-character subsequence
if is_subsequence(t_lower, s):
    print("Yes")
else:
    # Check for 2-character subsequence followed by 'X'
    if t[2] == 'X':
        t_first_two = t[:2].lower()
        if is_subsequence(t_first_two, s):
            print("Yes")
        else:
            print("No")
    else:
        print("No")