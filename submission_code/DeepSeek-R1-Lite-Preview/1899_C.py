def max_sum_alternating_parity_subarray(a):
    current_max_even = float('-inf')
    current_max_odd = float('-inf')
    global_max = float('-inf')
    
    for num in a:
        if num % 2 == 0:
            current_max_even = max(num, current_max_odd + num)
        else:
            current_max_odd = max(num, current_max_even + num)
        
        current_global = max(current_max_even, current_max_odd)
        if current_global > global_max:
            global_max = current_global
    
    return global_max

def main():
    import sys
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        result = max_sum_alternating_parity_subarray(a)
        print(result)

if __name__ == "__main__":
    main()