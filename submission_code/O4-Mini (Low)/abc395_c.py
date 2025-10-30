import sys
import threading

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    last_pos = {}
    min_len = float('inf')

    for i, x in enumerate(A):
        if x in last_pos:
            length = i - last_pos[x] + 1
            if length < min_len:
                min_len = length
        last_pos[x] = i

    if min_len == float('inf'):
        print(-1)
    else:
        print(min_len)

if __name__ == "__main__":
    main()