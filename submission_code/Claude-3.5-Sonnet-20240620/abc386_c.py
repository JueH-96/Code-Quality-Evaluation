# YOUR CODE HERE
def can_transform(s, t):
    if s == t:
        return True
    
    if abs(len(s) - len(t)) > 1:
        return False
    
    if len(s) == len(t):
        diff = sum(1 for a, b in zip(s, t) if a != b)
        return diff <= 1
    
    if len(s) < len(t):
        s, t = t, s
    
    i = j = 0
    diff = 0
    while i < len(s) and j < len(t):
        if s[i] != t[j]:
            diff += 1
            if diff > 1:
                return False
            i += 1
        else:
            i += 1
            j += 1
    
    return True

# Read input
k = int(input())
s = input().strip()
t = input().strip()

# Solve and output
result = "Yes" if can_transform(s, t) else "No"
print(result)