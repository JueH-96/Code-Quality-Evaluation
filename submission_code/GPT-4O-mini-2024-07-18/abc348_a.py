# YOUR CODE HERE
def penalty_kicks(N):
    results = []
    for i in range(1, N + 1):
        if i % 3 == 0:
            results.append('x')
        else:
            results.append('o')
    return ''.join(results)

N = int(input().strip())
print(penalty_kicks(N))