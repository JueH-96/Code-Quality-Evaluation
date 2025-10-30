# YOUR CODE HERE
A, B = map(int, input().split())
results = set()
results.add(2*A - B)
results.add(2*B - A)
if (A + B) % 2 == 0:
    results.add((A + B) // 2)
print(len(results))