# YOUR CODE HERE
def solve():
    n = int(input())
    p = list(map(int, input().split()))

    def operate(arr, k):
        new_arr = arr[:]
        if k >= 2:
            new_arr[:k-1] = sorted(new_arr[:k-1])
        if k <= n - 1:
            new_arr[k:] = sorted(new_arr[k:])
        return new_arr

    q = [(p, 0)]
    visited = {tuple(p)}
    
    while q:
        curr_p, dist = q.pop(0)
        if all(curr_p[i] == i + 1 for i in range(n)):
            print(dist)
            return

        for k in range(1, n + 1):
            next_p = operate(curr_p, k)
            if tuple(next_p) not in visited:
                visited.add(tuple(next_p))
                q.append((next_p, dist + 1))

t = int(input())
for _ in range(t):
    solve()