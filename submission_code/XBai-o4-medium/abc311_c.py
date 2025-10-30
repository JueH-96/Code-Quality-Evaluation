import sys

def main():
    n = int(sys.stdin.readline())
    a_list = list(map(int, sys.stdin.readline().split()))
    A = [0] * (n + 1)
    for i in range(n):
        A[i+1] = a_list[i]
    
    visited = [False] * (n + 1)
    
    for node in range(1, n + 1):
        if not visited[node]:
            path = []
            pos = {}
            current = node
            while True:
                if visited[current]:
                    break
                visited[current] = True
                path.append(current)
                pos[current] = len(path) - 1
                current = A[current]
            if current in pos:
                idx = pos[current]
                cycle = path[idx:]
                print(len(cycle))
                print(' '.join(map(str, cycle)))
                sys.exit()
                
if __name__ == "__main__":
    main()