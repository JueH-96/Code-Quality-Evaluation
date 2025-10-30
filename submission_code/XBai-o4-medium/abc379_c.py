import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    x_list = list(map(int, sys.stdin.readline().split()))
    a_list = list(map(int, sys.stdin.readline().split()))
    
    total_stones = sum(a_list)
    if total_stones != n:
        print(-1)
        return
    
    pairs = sorted(zip(x_list, a_list), key=lambda p: p[0])
    
    prev_end = 0
    total_ops = 0
    
    for x, a in pairs:
        start = max(prev_end + 1, x)
        end = start + a - 1
        if end > n:
            print(-1)
            return
        contribution = a * (start - x) + (a * (a - 1)) // 2
        total_ops += contribution
        prev_end = end
    
    print(total_ops)

if __name__ == "__main__":
    main()