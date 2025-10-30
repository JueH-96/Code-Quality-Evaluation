import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    x = []
    for _ in range(n):
        x.append(list(map(int, sys.stdin.readline().split())))
    
    # Initialize sums for each nutrient
    sums = [0] * m
    for i in range(n):
        for j in range(m):
            sums[j] += x[i][j]
    
    # Check if all nutrients meet the requirement
    ok = True
    for j in range(m):
        if sums[j] < a[j]:
            ok = False
            break
    
    print("Yes" if ok else "No")

if __name__ == "__main__":
    main()