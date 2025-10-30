import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A_list = list(map(int, data[1:1+n]))
    A = [0] + A_list
    visited = [0] * (n + 1)

    for i in range(1, n + 1):
        if visited[i] != 0:
            continue
        path = []
        cur = i
        while True:
            visited[cur] = 1
            path.append(cur)
            nxt = A[cur]
            if visited[nxt] == 0:
                cur = nxt
            elif visited[nxt] == 1:
                idx = path.index(nxt)
                cycle = path[idx:]
                print(len(cycle))
                print(" ".join(map(str, cycle)))
                return
            else:
                break
        for node in path:
            visited[node] = 2

if __name__ == "__main__":
    main()