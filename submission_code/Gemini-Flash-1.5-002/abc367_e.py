# YOUR CODE HERE
n, k = map(int, input().split())
x = list(map(int, input().split()))
a = list(map(int, input().split()))

x = [i - 1 for i in x]  # Adjust indices to be 0-based

b = a[:]
history = []
for _ in range(min(k, n + 10)):
    if b in history:
        first_occurrence = history.index(b)
        remaining_iterations = k - len(history)
        cycle_length = len(history) - first_occurrence
        index_in_cycle = remaining_iterations % cycle_length
        b = history[first_occurrence + index_in_cycle]
        break
    history.append(b[:])
    c = [0] * n
    for i in range(n):
        c[i] = b[x[i]]
    b = c

print(*b)