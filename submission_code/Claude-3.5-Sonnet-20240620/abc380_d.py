# YOUR CODE HERE
def transform(s):
    return ''.join(c.lower() if c.isupper() else c.upper() for c in s)

def get_char_at_position(s, k):
    length = len(s)
    double_length = 2 * length
    
    if k <= length:
        return s[k - 1]
    
    k = (k - 1) % double_length + 1
    
    if k <= length:
        return s[k - 1]
    else:
        return transform(s)[k - length - 1]

s = input().strip()
q = int(input())
queries = list(map(int, input().split()))

results = []
for k in queries:
    results.append(get_char_at_position(s, k))

print(' '.join(results))