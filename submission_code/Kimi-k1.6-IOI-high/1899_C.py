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
        a = list(map(int, data[idx:idx+n]))
        idx += n
        
        prev_even = prev_odd = float('-inf')
        global_max = float('-inf')
        
        for num in a:
            if num % 2 == 0:
                current_even = max(num, prev_odd + num) if prev_odd != float('-inf') else num
                current_odd = float('-inf')
            else:
                current_odd = max(num, prev_even + num) if prev_even != float('-inf') else num
                current_even = float('-inf')
            
            current_max = max(current_even, current_odd)
            if current_max > global_max:
                global_max = current_max
            
            prev_even, prev_odd = current_even, current_odd
        
        print(global_max)

if __name__ == "__main__":
    main()