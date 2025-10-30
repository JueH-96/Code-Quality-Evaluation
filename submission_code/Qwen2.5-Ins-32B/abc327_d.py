import sys

def is_good_pair(N, M, A, B):
    color = [None] * (N + 1)
    for i in range(M):
        a, b = A[i], B[i]
        if color[a] is None and color[b] is None:
            color[a], color[b] = 0, 1
        elif color[a] is None:
            color[a] = 1 - color[b]
        elif color[b] is None:
            color[b] = 1 - color[a]
        elif color[a] == color[b]:
            return False
    return True

if __name__ == "__main__":
    input = sys.stdin.read
    N, M, *AB = map(int, input().split())
    A, B = AB[:M], AB[M:]
    if is_good_pair(N, M, A, B):
        print("Yes")
    else:
        print("No")