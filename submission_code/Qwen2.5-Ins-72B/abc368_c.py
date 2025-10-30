# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:]))
    
    T = 0
    while H:
        if T % 3 == 2:
            H[0] -= 3
        else:
            H[0] -= 1
        if H[0] <= 0:
            H.pop(0)
        T += 1
    print(T)

if __name__ == "__main__":
    solve()