import sys
input = sys.stdin.readline

def main():
    n = int(input())
    p = list(map(int, input().split()))
    p = [0] + p + [n+1]
    right = [0] * (n+2)
    for i in range(n, 0, -1):
        right[i] = min(right[i+1], p[i])
    ans = 0
    for i in range(1, n+1):
        if p[i] > i and right[p[i]] < p[i-1]:
            ans += 1
    print(ans)
    
    
for _ in range(int(input())):
    main()