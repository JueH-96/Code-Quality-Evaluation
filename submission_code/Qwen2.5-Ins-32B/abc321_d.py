# YOUR CODE HERE
import sys

def calculate_total_price(N, M, P, A, B):
    total_price = 0
    A.sort()
    B.sort()
    for a in A:
        for b in B:
            total_price += min(a + b, P)
    return total_price

def main():
    input = sys.stdin.read
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    P = int(data[index])
    index += 1
    A = list(map(int, data[index:index+N]))
    index += N
    B = list(map(int, data[index:index+M]))
    index += M
    print(calculate_total_price(N, M, P, A, B))

if __name__ == "__main__":
    main()