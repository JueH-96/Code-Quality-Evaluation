# YOUR CODE HERE

import sys

def solve(N, H):
    c = [0]*N
    stack = []
    for i in range(N):
        while stack and H[stack[-1]] < H[i]:
            stack.pop()
        c[i] = len(stack)
        stack.append(i)
    return c

def main():
    N = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))
    c = solve(N, H)
    print(' '.join(map(str, c)))

if __name__ == "__main__":
    main()