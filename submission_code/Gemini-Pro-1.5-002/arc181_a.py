# YOUR CODE HERE
def solve():
    n = int(input())
    p = list(map(int, input().split()))

    def is_sorted(arr):
        return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

    def check(arr):
        return all(arr[i] == i + 1 for i in range(len(arr)))

    q = [(p, 0)]
    visited = {tuple(p)}

    while q:
        curr_p, steps = q.pop(0)
        if check(curr_p):
            print(steps)
            return

        for k in range(1, n + 1):
            next_p = list(curr_p)
            if k >= 2:
                next_p[:k-1] = sorted(next_p[:k-1])
            if k <= n - 1:
                next_p[k:] = sorted(next_p[k:])
            
            if tuple(next_p) not in visited:
                visited.add(tuple(next_p))
                q.append((next_p, steps + 1))

t = int(input())
for _ in range(t):
    solve()