# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    a = list(map(int, data[3:3+N]))
    b = list(map(int, data[3+N:3+N+M]))
    
    forbidden = set()
    for i in range(L):
        c, d = map(int, data[3+N+M+2*i:3+N+M+2*i+2])
        forbidden.add((c-1, d-1))
    
    max_price = 0
    for i in range(N):
        for j in range(M):
            if (i, j) not in forbidden:
                max_price = max(max_price, a[i] + b[j])
    
    print(max_price)

if __name__ == "__main__":
    main()