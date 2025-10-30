import sys

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    visited = [False] * (n + 1)
    
    for i in range(1, n + 1):
        if not visited[i]:
            path = []
            pos = dict()
            current = i
            while True:
                if visited[current]:
                    if current in pos:
                        cycle = path[pos[current]:]
                        print(len(cycle))
                        print(' '.join(map(str, cycle)))
                        return
                    else:
                        break
                else:
                    visited[current] = True
                    path.append(current)
                    pos[current] = len(path) - 1
                    current = A[current - 1]

if __name__ == '__main__':
    main()