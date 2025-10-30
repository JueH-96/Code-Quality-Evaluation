import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    s_t = []
    for _ in range(n-1):
        s, t = map(int, sys.stdin.readline().split())
        s_t.append((s, t))
    
    for i in range(n-1):
        s, t = s_t[i]
        cnt = a[i] // s
        a[i] -= cnt * s
        a[i+1] += cnt * t
    
    print(a[-1])

if __name__ == "__main__":
    main()