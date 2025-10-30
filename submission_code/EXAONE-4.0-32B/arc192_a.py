import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if all(a == 1 for a in A):
        print("Yes")
        return
        
    if n % 2 != 0:
        print("No")
        return
        
    dp1 = [[False] * 2 for _ in range(2)]
    if A[0] == 0:
        dp1[0][1] = True
    else:
        dp1[0][0] = True
        dp1[0][1] = True
        
    for i in range(1, n-1):
        new_dp = [[False] * 2 for _ in range(2)]
        for a in range(2):
            for b in range(2):
                if not dp1[a][b]:
                    continue
                for c in range(2):
                    if b == 1 and c == 1:
                        continue
                    if A[i] == 0 and b == 0 and c == 0:
                        continue
                    new_dp[b][c] = True
        dp1 = new_dp
        
    found1 = False
    for a in range(2):
        for b in range(2):
            if not dp1[a][b]:
                continue
            if A[n-1] == 0 and b == 0:
                continue
            found1 = True
            
    if found1:
        print("Yes")
        return
        
    dp2 = [[False] * 2 for _ in range(2)]
    dp2[1][0] = True
    
    for i in range(1, n-1):
        new_dp = [[False] * 2 for _ in range(2)]
        for a in range(2):
            for b in range(2):
                if not dp2[a][b]:
                    continue
                for c in range(2):
                    if b == 1 and c == 1:
                        continue
                    if A[i] == 0 and b == 0 and c == 0:
                        continue
                    new_dp[b][c] = True
        dp2 = new_dp
        
    found2 = False
    for a in range(2):
        for b in range(2):
            if not dp2[a][b]:
                continue
            if b == 1:
                continue
            found2 = True
            
    if found2:
        print("Yes")
        return
        
    print("No")

if __name__ == "__main__":
    main()