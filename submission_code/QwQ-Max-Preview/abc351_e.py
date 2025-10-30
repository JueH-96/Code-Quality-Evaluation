import sys

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    even = []
    odd = []
    idx = 1
    for _ in range(n):
        x = int(input[idx])
        y = int(input[idx + 1])
        idx += 2
        s = x + y
        if s % 2 == 0:
            even.append((x, y))
        else:
            odd.append((x, y))
    
    def compute_contribution(points):
        m = len(points)
        if m < 2:
            return 0
        u = [x + y for x, y in points]
        v = [x - y for x, y in points]
        
        sorted_u = sorted(u)
        prefix = [0] * (m + 1)
        for i in range(m):
            prefix[i + 1] = prefix[i] + sorted_u[i]
        sum_u = 0
        for i in range(m):
            sum_u += sorted_u[i] * i - prefix[i]
        
        sorted_v = sorted(v)
        prefix_v = [0] * (m + 1)
        for i in range(m):
            prefix_v[i + 1] = prefix_v[i] + sorted_v[i]
        sum_v = 0
        for i in range(m):
            sum_v += sorted_v[i] * i - prefix_v[i]
        
        return (sum_u + sum_v) // 2
    
    even_contribution = compute_contribution(even)
    odd_contribution = compute_contribution(odd)
    print(even_contribution + odd_contribution)

if __name__ == '__main__':
    main()