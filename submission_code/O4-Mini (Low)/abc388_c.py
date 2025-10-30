import sys
import threading

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    ans = 0
    ptr = 0
    for j in range(n):
        half = A[j] // 2
        # advance ptr while A[ptr] <= half and ptr < j
        while ptr < j and A[ptr] <= half:
            ptr += 1
        ans += ptr

    print(ans)

if __name__ == "__main__":
    main()