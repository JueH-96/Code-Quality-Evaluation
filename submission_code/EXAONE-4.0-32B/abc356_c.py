import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
        
    first_line = data[0].split()
    n = int(first_line[0])
    m = int(first_line[1])
    k = int(first_line[2])
    
    tests = []
    for i in range(1, m+1):
        parts = data[i].split()
        c = int(parts[0])
        keys = list(map(int, parts[1:1+c]))
        r = parts[-1]
        
        test_mask = 0
        for key in keys:
            test_mask |= (1 << (key-1))
            
        tests.append((test_mask, r))
        
    total_masks = 1 << n
    popcount_arr = [0] * total_masks
    for mask_val in range(total_masks):
        popcount_arr[mask_val] = bin(mask_val).count('1')
    
    count_valid = 0
    for candidate in range(total_masks):
        valid = True
        for (test_mask, r) in tests:
            inter = candidate & test_mask
            real_count = popcount_arr[inter]
            if r == 'o':
                if real_count < k:
                    valid = False
                    break
            elif r == 'x':
                if real_count >= k:
                    valid = False
                    break
                    
        if valid:
            count_valid += 1
            
    print(count_valid)

if __name__ == "__main__":
    main()