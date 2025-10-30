# YOUR CODE HERE
def ccw(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def intersect(p1, q1, p2, q2):
    return ccw(p1, q1, p2) * ccw(p1, q1, q2) <= 0 and ccw(p2, q2, p1) * ccw(p2, q2, q1) <= 0

def solve(n, p, q):
    used = [False] * n
    result = [0] * n

    def dfs(i):
        if i == n:
            return True
        for j in range(n):
            if used[j]:
                continue
            valid = True
            for k in range(i):
                if intersect(p[i], q[j], p[k], q[result[k]]):
                    valid = False
                    break
            if valid:
                used[j] = True
                result[i] = j
                if dfs(i + 1):
                    return True
                used[j] = False
        return False

    if dfs(0):
        return [r + 1 for r in result]
    return [-1]

n = int(input())
p = [tuple(map(int, input().split())) for _ in range(n)]
q = [tuple(map(int, input().split())) for _ in range(n)]

result = solve(n, p, q)
print(*result)