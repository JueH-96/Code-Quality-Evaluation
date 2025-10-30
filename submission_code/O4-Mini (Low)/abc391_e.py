import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    N = int(data[0])
    A_str = data[1].strip()
    A = list(map(int, A_str))
    
    # Precompute valid combinations of child outputs (x,y,z) whose majority is 0 or 1
    valid = {0: [], 1: []}
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                s = x + y + z
                maj = 1 if s >= 2 else 0
                valid[maj].append((x,y,z))
    
    # Recursive function: solve segment A[l:r], where length = 3^k, returns dp0,dp1
    def solve(l, r):
        length = r - l
        if length == 1:
            bit = A[l]
            # To make this leaf 0 costs (bit != 0), to make 1 costs (bit != 1)
            return (0 if bit == 0 else 1, 0 if bit == 1 else 1)
        # split into three equal parts
        third = length // 3
        d10, d11 = solve(l, l + third)
        d20, d21 = solve(l + third, l + 2*third)
        d30, d31 = solve(l + 2*third, r)
        dp0 = 10**18
        dp1 = 10**18
        # Compute cost to make this node 0 or 1
        for x,y,z in valid[0]:
            cost = (d10 if x==0 else d11) + (d20 if y==0 else d21) + (d30 if z==0 else d31)
            if cost < dp0:
                dp0 = cost
        for x,y,z in valid[1]:
            cost = (d10 if x==0 else d11) + (d20 if y==0 else d21) + (d30 if z==0 else d31)
            if cost < dp1:
                dp1 = cost
        return (dp0, dp1)
    
    dp0, dp1 = solve(0, len(A))
    # Original value at root must be the one with zero cost
    if dp0 == 0:
        orig = 0
    else:
        orig = 1
    # We need to flip to the opposite
    answer = dp1 if orig == 0 else dp0
    print(answer)

if __name__ == "__main__":
    main()