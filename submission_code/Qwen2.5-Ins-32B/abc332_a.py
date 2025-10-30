# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    K = int(data[2])
    
    total_price = 0
    for i in range(N):
        P = int(data[3 + 2 * i])
        Q = int(data[4 + 2 * i])
        total_price += P * Q
    
    if total_price < S:
        total_price += K
    
    print(total_price)

if __name__ == "__main__":
    main()