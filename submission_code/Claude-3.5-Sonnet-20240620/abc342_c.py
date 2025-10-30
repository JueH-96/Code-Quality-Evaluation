# YOUR CODE HERE
def replace_chars(s, operations):
    char_map = {}
    for c, d in operations:
        if c not in char_map:
            char_map[c] = c
        char_map[c] = d

    final_map = {}
    for c in 'abcdefghijklmnopqrstuvwxyz':
        curr = c
        while curr in char_map and char_map[curr] != curr:
            curr = char_map[curr]
        final_map[c] = curr

    return ''.join(final_map[c] for c in s)

N = int(input())
S = input().strip()
Q = int(input())

operations = []
for _ in range(Q):
    c, d = input().split()
    operations.append((c, d))

result = replace_chars(S, operations)
print(result)