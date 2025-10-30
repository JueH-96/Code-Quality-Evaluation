import sys
from itertools import combinations

MOD = 10**9 + 7

def main():
    grid = []
    for _ in range(3):
        line = list(map(int, sys.stdin.readline().split()))
        grid.append(line)
    
    lines = []
    for i in range(3):
        lines.append(grid[i])
    for i in range(3):
        col = [grid[j][i] for j in range(3)]
        lines.append(col)
    diag1 = [grid[0][0], grid[1][1], grid[2][2]]
    lines.append(diag1)
    diag2 = [grid[0][2], grid[1][1], grid[2][0]]
    lines.append(diag2)
    
    n = 9
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i % MOD
    
    total = fact[n]
    
    forbidden = []
    for line in lines:
        line_set = set(line)
        if len(line_set) != 3:
            forbidden.append(0)
        else:
            count = 0
            for a in line_set:
                for b in line_set:
                    if a == b:
                        continue
                    for c in line_set:
                        if a == c:
                            continue
                        if b == c:
                            continue
                        count += 1
            count = count * (fact[n - 3] * fact[3]) % MOD
            forbidden.append(count)
    
    k = len(lines)
    inclusion_exclusion = 0
    for mask in range(1, 1 << k):
        bits = bin(mask).count('1')
        sign = (-1) ** bits
        count = 0
        for i in range(k):
            if (mask >> i) & 1:
                count += forbidden[i]
        if count == 0:
            continue
        term = pow(-1, bits) * comb(k, bits) * (fact[n] // (fact[n - count] * product))
        inclusion_exclusion += term
    
    valid = (total - inclusion_exclusion) % MOD
    prob = valid * pow(total, MOD-2, MOD) % MOD
    print("{0:.15f}".format(prob))

if __name__ == '__main__':
    main()