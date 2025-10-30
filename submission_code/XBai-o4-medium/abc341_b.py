import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    s_list = []
    t_list = []
    for _ in range(n-1):
        s, t = map(int, sys.stdin.readline().split())
        s_list.append(s)
        t_list.append(t)
    
    for i in range(n-1):
        s = s_list[i]
        t = t_list[i]
        q = a[i] // s
        a[i] -= q * s
        a[i+1] += q * t
    
    print(a[-1])

if __name__ == "__main__":
    main()