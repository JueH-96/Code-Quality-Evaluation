import sys

def main():
    n = int(sys.stdin.readline())
    l = []
    r = []
    for _ in range(n):
        a, b = map(int, sys.stdin.readline().split())
        l.append(a)
        r.append(b)
    
    sum_L = sum(l)
    sum_R = sum(r)
    
    if sum_L > 0 or sum_R < 0:
        print("No")
        return
    
    delta = 0 - sum_L
    current_delta = delta
    x = list(l)
    
    for i in range(n):
        available = r[i] - x[i]
        add = min(current_delta, available)
        x[i] += add
        current_delta -= add
        if current_delta == 0:
            break
    
    print("Yes")
    print(' '.join(map(str, x)))

if __name__ == "__main__":
    main()