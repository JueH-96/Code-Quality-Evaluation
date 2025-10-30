import sys
input = sys.stdin.read

def find_permutation(N, P, Q):
    def intersect(p1, q1, p2, q2):
        def ccw(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, q1, p2) * ccw(p1, q1, q2) < 0 and ccw(p2, q2, p1) * ccw(p2, q2, q1) < 0

    def backtrack(idx, used, path):
        if idx == N:
            return path
        for i in range(N):
            if not used[i]:
                valid = True
                for j in range(idx):
                    if intersect(P[j], Q[path[j]], P[idx], Q[i]):
                        valid = False
                        break
                if valid:
                    used[i] = True
                    path[idx] = i
                    result = backtrack(idx + 1, used, path)
                    if result:
                        return result
                    used[i] = False
        return None

    used = [False] * N
    path = [0] * N
    result = backtrack(0, used, path)

    if result is None:
        print(-1)
    else:
        print(' '.join(map(str, [x + 1 for x in result])))

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    P = []
    Q = []
    for i in range(N):
        A, B = int(data[index]), int(data[index + 1])
        index += 2
        P.append((A, B))
    for i in range(N):
        C, D = int(data[index]), int(data[index + 1])
        index += 2
        Q.append((C, D))

    find_permutation(N, P, Q)

if __name__ == "__main__":
    main()