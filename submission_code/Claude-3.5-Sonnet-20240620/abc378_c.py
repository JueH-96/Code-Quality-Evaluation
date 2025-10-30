# YOUR CODE HERE
def find_sequence_b(n, a):
    last_seen = {}
    b = []
    for i in range(n):
        if a[i] in last_seen:
            b.append(last_seen[a[i]] + 1)
        else:
            b.append(-1)
        last_seen[a[i]] = i
    return b

# Read input
n = int(input())
a = list(map(int, input().split()))

# Solve and print output
result = find_sequence_b(n, a)
print(*result)