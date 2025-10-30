import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx + n]))
        idx += n
        
        even_max = -float('inf')
        odd_max = -float('inf')
        max_sum = -float('inf')
        
        for num in a:
            if num % 2 == 0:
                current_even = max(num, odd_max + num)
                current_odd = -float('inf')
            else:
                current_odd = max(num, even_max + num)
                current_even = -float('inf')
            
            even_max = current_even
            odd_max = current_odd
            
            if current_even != -float('inf'):
                if current_even > max_sum:
                    max_sum = current_even
            if current_odd != -float('inf'):
                if current_odd > max_sum:
                    max_sum = current_odd
        
        print(max_sum if max_sum != -float('inf') else 0)

if __name__ == '__main__':
    main()