from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    positions = defaultdict(list)
    for i in range(n):
        x = a[i]
        positions[x].append(i + 1)  # Store 1-based index
    
    total = 0
    
    for x in positions:
        pos = positions[x]
        m = len(pos)
        if m < 2:
            continue
        
        sum1 = 0
        for b_code in range(m):
            term = b_code * (pos[b_code] - (b_code + 1))
            sum1 += term
        
        sum2 = 0
        for a_code in range(m):
            term = (m - (a_code + 1)) * (pos[a_code] - (a_code + 1))
            sum2 += term
        
        contribution = sum1 - sum2
        total += contribution
    
    print(total)

if __name__ == '__main__':
    main()