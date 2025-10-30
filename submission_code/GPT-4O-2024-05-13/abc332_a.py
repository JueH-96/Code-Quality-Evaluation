# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    S = int(data[1])
    K = int(data[2])
    
    total_price = 0
    index = 3
    for _ in range(N):
        P = int(data[index])
        Q = int(data[index + 1])
        total_price += P * Q
        index += 2
    
    if total_price >= S:
        print(total_price)
    else:
        print(total_price + K)

if __name__ == "__main__":
    main()