import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    XYZ_line = data[0].split()
    X = int(XYZ_line[0])
    Y = int(XYZ_line[1])
    Z = int(XYZ_line[2])
    S = data[1].strip()
    
    dp0 = 0
    dp1 = 10**18
    
    for c in S:
        next_dp0 = 10**18
        next_dp1 = 10**18
        
        if c == 'a':
            next_dp0 = min(next_dp0, dp0 + X)
            next_dp1 = min(next_dp1, dp0 + Z + Y)
        else:
            next_dp0 = min(next_dp0, dp0 + Y)
            next_dp1 = min(next_dp1, dp0 + Z + X)
            
        if c == 'a':
            next_dp1 = min(next_dp1, dp1 + Y)
            next_dp0 = min(next_dp0, dp1 + Z + X)
        else:
            next_dp1 = min(next_dp1, dp1 + X)
            next_dp0 = min(next_dp0, dp1 + Z + Y)
        
        dp0, dp1 = next_dp0, next_dp1
        
    print(min(dp0, dp1))

if __name__ == "__main__":
    main()