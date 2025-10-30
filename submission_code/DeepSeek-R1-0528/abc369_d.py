import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    dp0 = 0
    dp1 = -10**20
    
    for a in A:
        new_dp0 = max(dp0, dp1 + 2*a)
        new_dp1 = max(dp1, dp0 + a)
        dp0, dp1 = new_dp0, new_dp1
        
    print(max(dp0, dp1))

if __name__ == "__main__":
    main()