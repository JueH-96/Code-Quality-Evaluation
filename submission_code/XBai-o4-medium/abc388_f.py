import sys

def main():
    n, m, a, b = map(int, sys.stdin.readline().split())
    bad = []
    for _ in range(m):
        l, r = map(int, sys.stdin.readline().split())
        bad.append((l, r))
    
    good = []
    prev = 1
    for l, r in bad:
        if prev <= l - 1:
            good.append((prev, l - 1))
        prev = r + 1
    # Add the last good interval
    good.append((prev, n))
    
    max_reach = 1
    for s, e in good:
        if max_reach >= s - b:
            temp = e + b
            max_reach = max(max_reach, temp)
            if max_reach > n:
                max_reach = n
    
    if max_reach >= n:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()