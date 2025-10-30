# YOUR CODE HERE
def solve(n, k, s):
    operations = 0
    i = 0
    while i < n:
        if s[i] == 'B':
            operations += 1
            i += k
        else:
            i += 1
    return operations

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    print(solve(n, k, s))
# YOUR CODE HERE