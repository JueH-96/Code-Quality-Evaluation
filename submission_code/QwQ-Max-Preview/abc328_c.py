import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    a = []
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            a.append(1)
        else:
            a.append(0)
    
    # Compute prefix sums
    prefix = [0] * (len(a) + 1)
    for i in range(len(a)):
        prefix[i + 1] = prefix[i] + a[i]
    
    # Process each query
    for _ in range(q):
        l, r = map(int, sys.stdin.readline().split())
        start = l - 1
        end = r - 2
        if start > end:
            print(0)
        else:
            print(prefix[end + 1] - prefix[start])

if __name__ == "__main__":
    main()