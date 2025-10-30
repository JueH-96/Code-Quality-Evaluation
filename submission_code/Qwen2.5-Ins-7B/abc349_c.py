# YOUR CODE HERE
s = input()
t = input()

def is_subsequence(s, t):
    it = iter(s)
    return all(c in it for c in t)

def check(s, t):
    if t[0] == t[1] == t[2]:
        return is_subsequence(s, t[0]*3)
    elif t[0] == t[1]:
        return is_subsequence(s, t[0] + t[2])
    elif t[1] == t[2]:
        return is_subsequence(s, t[0] + t[1])
    else:
        return is_subsequence(s, t[0]) and is_subsequence(s, t[1]) and is_subsequence(s, t[2])

print('Yes' if check(s, t) else 'No')