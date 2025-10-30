# YOUR CODE HERE
def is_subsequence(s, t):
    it = iter(s)
    return all(c in it for c in t)

def count_valid_pairs(n, t, strings):
    prefixes = [0] * (len(t) + 1)
    count = 0
    
    for s in strings:
        current = 0
        for i, c in enumerate(s):
            if current < len(t) and c == t[current]:
                current += 1
            if current == len(t):
                count += n
                break
        prefixes[current] += 1
    
    for i in range(len(t) - 1, -1, -1):
        prefixes[i] += prefixes[i + 1]
    
    for s in strings:
        current = 0
        for i, c in enumerate(s):
            if current < len(t) and c == t[current]:
                current += 1
        count += prefixes[len(t) - current]
    
    return count

n, t = input().split()
n = int(n)
strings = [input().strip() for _ in range(n)]

print(count_valid_pairs(n, t, strings))