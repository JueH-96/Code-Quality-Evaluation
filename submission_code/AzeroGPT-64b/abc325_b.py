def solve(N, W, X):
    attend = [0] * 25
    for i in range(N):
        start, end = max(0, 9 - X[i]), min(24, 18 - X[i])
        attend[start] += W[i]
        attend[end + 1] -= W[i]

    ans = mx = 0
    for i in range(24):
        mx += attend[i + 1]
        ans = max(ans, mx)
    return ans

if __name__ == '__main__':
    N = int(input())
    W, X = [], []
    for _ in range(N):
        w, x = map(int, input().split())
        W.append(w)
        X.append(x)
    print(solve(N, W, X))