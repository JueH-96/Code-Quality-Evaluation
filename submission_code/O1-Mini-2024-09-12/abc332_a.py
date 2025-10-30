# YOUR CODE HERE
import sys

def main():
    data = list(map(int, sys.stdin.read().split()))
    N, S, K = data[:3]
    total = 0
    for i in range(N):
        P, Q = data[3 + 2*i : 5 + 2*i]
        total += P * Q
    if total >= S:
        shipping = 0
    else:
        shipping = K
    print(total + shipping)

if __name__ == "__main__":
    main()