# YOUR CODE HERE
def can_be_equal(s, t):
    if s == t:
        return True
    if len(s) == len(t):
        diff = sum(1 for a, b in zip(s, t) if a != b)
        return diff == 1
    if len(s) + 1 == len(t):
        i = j = 0
        diff = 0
        while i < len(s) and j < len(t):
            if s[i] != t[j]:
                diff += 1
                j += 1
            else:
                i += 1
                j += 1
        return diff <= 1
    if len(s) == len(t) + 1:
        i = j = 0
        diff = 0
        while i < len(s) and j < len(t):
            if s[i] != t[j]:
                diff += 1
                i += 1
            else:
                i += 1
                j += 1
        return diff <= 1
    return False

n, t_prime = input().split()
n = int(n)

possible_t = []
for i in range(1, n + 1):
    s = input().strip()
    if can_be_equal(s, t_prime):
        possible_t.append(i)

print(len(possible_t))
print(*possible_t)