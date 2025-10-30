import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    a = list(map(int, data[2:2+n]))
    L = n - k
    a.sort()
    ans = 10**18
    for i in range(0, n - L + 1):
        diff = a[i+L-1] - a[i]
        if diff < ans:
            ans = diff
    print(ans)

if __name__ == "__main__":
    main()