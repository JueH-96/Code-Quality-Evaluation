import sys

def main():
    N = int(sys.stdin.readline())
    starts = [0] * (2 * N + 1)
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        a = min(A, B)
        b = max(A, B)
        starts[a] = b
    stack = []
    for x in range(1, 2 * N + 1):
        if starts[x] != 0:
            stack.append(starts[x])
        else:
            if not stack or stack[-1] != x:
                print("Yes")
                return
            stack.pop()
    print("No")

if __name__ == "__main__":
    main()