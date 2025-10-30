import sys

def main():
    n, x = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))
    p = list(map(int, sys.stdin.readline().split()))
    q = list(map(int, sys.stdin.readline().split()))
    
    s = set()
    
    for i in range(n):
        if a[i] > 0:
            current = i + 1
            visited = set()
            while True:
                if current == x:
                    break
                if current in visited:
                    print(-1)
                    return
                visited.add(current)
                s.add(current)
                current = p[current - 1]  # p is 1-based
        if b[i] > 0:
            current = i + 1
            visited = set()
            while True:
                if current == x:
                    break
                if current in visited:
                    print(-1)
                    return
                visited.add(current)
                s.add(current)
                current = q[current - 1]  # q is 1-based
    
    print(len(s))

if __name__ == "__main__":
    main()