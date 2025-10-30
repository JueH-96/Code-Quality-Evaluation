def solve():
    import sys
    data = sys.stdin.read().strip().split()
    
    N = int(data[0])
    S = int(data[1])
    K = int(data[2])
    
    total_price = 0
    idx = 3
    for _ in range(N):
        P = int(data[idx])
        Q = int(data[idx+1])
        idx += 2
        total_price += P * Q
    
    if total_price < S:
        total_price += K
    
    print(total_price)

def main():
    solve()

if __name__ == "__main__":
    main()