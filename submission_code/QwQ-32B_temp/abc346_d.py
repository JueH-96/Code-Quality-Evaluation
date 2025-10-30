import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    c = list(map(int, sys.stdin.readline().split()))
    
    # Initialize left array
    left = [[0] * 2 for _ in range(n)]
    left[0][0] = 0 if s[0] == '0' else c[0]
    left[0][1] = 0 if s[0] == '1' else c[0]
    
    for j in range(1, n):
        current = s[j]
        cost0 = c[j] if current != '0' else 0
        cost1 = c[j] if current != '1' else 0
        left[j][0] = left[j-1][1] + cost0
        left[j][1] = left[j-1][0] + cost1
    
    # Initialize right array
    right = [[0] * 2 for _ in range(n)]
    right[-1][0] = 0 if s[-1] == '0' else c[-1]
    right[-1][1] = 0 if s[-1] == '1' else c[-1]
    
    for j in range(n-2, -1, -1):
        current = s[j]
        cost0 = c[j] if current != '0' else 0
        cost1 = c[j] if current != '1' else 0
        right[j][0] = cost0 + right[j+1][1]
        right[j][1] = cost1 + right[j+1][0]
    
    min_total = float('inf')
    for i in range(n-1):
        for a in 0, 1:
            current = left[i][a] + right[i+1][a]
            if current < min_total:
                min_total = current
    
    print(min_total)

if __name__ == "__main__":
    main()