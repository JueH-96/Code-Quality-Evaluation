def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    
    # Function to return all positive divisors of n
    # in no particular order.
    def get_divisors(x):
        divs = []
        i = 1
        while i * i <= x:
            if x % i == 0:
                divs.append(i)
                if i != x // i:
                    divs.append(x // i)
            i += 1
        return divs

    idx = 1
    out = []
    for _ in range(t):
        n = int(input_data[idx])
        idx += 1
        arr = list(map(int, input_data[idx:idx+n]))
        idx += n
        
        # Edge case: if n == 1, there's only one truck for any k=1 => difference = 0
        if n == 1:
            out.append("0")
            continue
        
        # Compute prefix sums
        prefix_sums = [0] * (n+1)
        for i in range(n):
            prefix_sums[i+1] = prefix_sums[i] + arr[i]
        
        # Find all divisors of n
        divisors = get_divisors(n)
        
        max_diff = 0
        # For each divisor d, we group boxes in blocks of size d
        for d in divisors:
            block_count = n // d
            # Compute sums of each block
            mn = float('inf')
            mx = -float('inf')
            for i in range(1, block_count + 1):
                block_sum = prefix_sums[i*d] - prefix_sums[(i - 1)*d]
                if block_sum < mn:
                    mn = block_sum
                if block_sum > mx:
                    mx = block_sum
            diff = mx - mn
            if diff > max_diff:
                max_diff = diff
        
        out.append(str(max_diff))
    
    print("
".join(out))

# Do not forget to call main()
main()