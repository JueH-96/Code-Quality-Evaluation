from collections import deque

n = int(input())
s = input().strip()

def get_valid_ranges(seq):
    ranges = []
    n = len(seq)
    for i in range(n):
        balance = 0
        for j in range(i, n):
            if seq[j] == '(':
                balance += 1
            else:
                balance -= 1
            if balance == 0:
                ranges.append((i, j))
    return ranges

def flip_range(seq, start, end):
    result = list(seq)
    for i in range(start, end + 1):
        result[i] = ')' if seq[i] == '(' else '('
    return ''.join(result)

visited = {s}
queue = deque([s])

while queue:
    current = queue.popleft()
    ranges = get_valid_ranges(current)
    
    for start, end in ranges:
        new_seq = flip_range(current, start, end)
        if new_seq not in visited:
            visited.add(new_seq)
            queue.append(new_seq)

print(len(visited))