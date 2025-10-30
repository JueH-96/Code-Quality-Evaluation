import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:]))
    
    min_cost_with_coupon = Q + min(D)
    min_cost = min(P, min_cost_with_coupon)
    
    print(min_cost)

if __name__ == "__main__":
    main()