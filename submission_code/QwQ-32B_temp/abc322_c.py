import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    ans = [0] * (n + 1)  # 1-based indexing

    for k in range(m):
        current = a[k]
        if k == 0:
            start = 1
        else:
            start = a[k-1] + 1
        end = current
        # Generate the list of values for this interval
        values = [current - i for i in range(start, end + 1)]
        # Assign to the slice
        ans[start:end+1] = values

    for i in range(1, n+1):
        print(ans[i])

if __name__ == "__main__":
    main()