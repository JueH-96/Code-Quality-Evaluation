from collections import defaultdict

def count_pairs(a):
    n = len(a)
    count = 0
    notes = defaultdict(list)
    
    for i in range(n):
        b = 2 ** a[i]
        notes[b].append(i)
    
    for b, indices in notes.items():
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                if b ** indices[j] == indices[i] ** b:
                    count += 1
    
    return count

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(count_pairs(a))