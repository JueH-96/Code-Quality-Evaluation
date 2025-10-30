import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    results = []
    
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx + n]))
        idx += n
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]
        
        max_diff = 0
        
        for k in range(1, n + 1):
            if n % k != 0:
                continue
            m = n // k
            current_max = -float('inf')
            current_min = float('inf')
            
            for j in range(m):
                start = j * k
                end = start + k
                s = prefix[end] - prefix[start]
                if s > current_max:
                    current_max = s
                if s < current_min:
                    current_min = s
            
            diff = current_max - current_min
            if diff > max_diff:
                max_diff = diff
        
        results.append(max_diff)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()