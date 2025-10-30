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
        
        max_sum = -float('inf')
        prev_even = -float('inf')
        prev_odd = -float('inf')
        
        for num in a:
            if num % 2 == 0:
                current_even = max(num, prev_odd + num)
                current_odd = -float('inf')
            else:
                current_odd = max(num, prev_even + num)
                current_even = -float('inf')
            
            if current_even > max_sum:
                max_sum = current_even
            if current_odd > max_sum:
                max_sum = current_odd
            
            prev_even, prev_odd = current_even, current_odd
        
        results.append(str(max_sum))
    
    print('
'.join(results))

if __name__ == "__main__":
    main()