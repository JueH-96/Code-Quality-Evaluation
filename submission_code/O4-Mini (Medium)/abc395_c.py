import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    last_occ = {}
    ans = n + 1

    for i, x in enumerate(A):
        if x in last_occ:
            length = i - last_occ[x] + 1
            if length < ans:
                ans = length
        last_occ[x] = i

    if ans == n + 1:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()