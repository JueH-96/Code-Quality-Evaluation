import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+n]))
    A.sort()
    r = 0
    ans = 0
    for l in range(n):
        while r < n and A[r] < A[l] + M:
            r += 1
        count = r - l
        if count > ans:
            ans = count
    print(ans)

if __name__ == '__main__':
    main()