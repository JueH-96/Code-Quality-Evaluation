def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    pos = 1

    # A helper function to find all divisors of a positive integer n
    def find_divisors(x):
        divs = []
        i = 1
        while i * i <= x:
            if x % i == 0:
                divs.append(i)
                if i * i != x:
                    divs.append(x // i)
            i += 1
        return divs

    out = []
    for _ in range(t):
        n = int(input_data[pos]); pos += 1
        arr = list(map(int, input_data[pos:pos+n]))
        pos += n

        # Quick handle for n=1
        if n == 1:
            # Only one truck is possible, difference = 0
            out.append("0")
            continue

        # Prefix sums
        prefix = [0] * (n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + arr[i]

        divisors = find_divisors(n)
        best_diff = 0

        # For each divisor d of n
        for d in divisors:
            # We'll have n/d trucks, each with consecutive d boxes
            # block_sum_i = prefix[i*d] - prefix[(i-1)*d], i=1..(n//d)
            block_min = float('inf')
            block_max = float('-inf')
            # compute sums in O(n/d)
            step_count = n // d
            # For i in [0..step_count-1], block i sum => prefix[(i+1)*d]-prefix[i*d]
            start = 0
            for i in range(step_count):
                s = prefix[(i+1)*d] - prefix[i*d]
                if s < block_min:
                    block_min = s
                if s > block_max:
                    block_max = s
            local_diff = block_max - block_min
            if local_diff > best_diff:
                best_diff = local_diff

        out.append(str(best_diff))

    print("
".join(out))

# Call main function
main()