# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    def check(arr, k):
        prod = 1
        for x in arr:
            prod = (prod * x) % k
        return prod == 0

    q = [(a, 0)]
    visited = {tuple(a)}
    while q:
        curr_a, curr_ops = q.pop(0)
        if check(curr_a, k):
            print(curr_ops)
            return

        for i in range(n):
            next_a = curr_a[:]
            next_a[i] += 1
            if tuple(next_a) not in visited:
                visited.add(tuple(next_a))
                q.append((next_a, curr_ops + 1))

t = int(input())
for _ in range(t):
    solve()