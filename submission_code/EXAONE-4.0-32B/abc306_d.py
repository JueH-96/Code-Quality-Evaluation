def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    courses = []
    for i in range(1, n + 1):
        line = data[i].split()
        x = int(line[0])
        y = int(line[1])
        courses.append((x, y))
    
    dp0 = 0
    dp1 = -10**18
    
    for x, y in courses:
        next_dp0 = dp0
        next_dp1 = dp1
        
        if x == 0:
            candidate0 = dp0 + y
            candidate1 = dp1 + y
            next_dp0 = max(next_dp0, candidate0, candidate1)
        else:
            candidate0 = dp0 + y
            next_dp1 = max(next_dp1, candidate0)
        
        dp0, dp1 = next_dp0, next_dp1
    
    print(max(dp0, dp1))

if __name__ == "__main__":
    main()