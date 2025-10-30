def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    # We'll process each test case one by one
    answers = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        
        # Prefix sums for the array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]
        
        # Find all divisors of n
        divisors = []
        d = 1
        # We only need to check up to sqrt(n) and add pairs
        while d * d <= n:
            if n % d == 0:
                divisors.append(d)
                if d * d != n:
                    divisors.append(n // d)
            d += 1
        
        best_diff = 0
        
        # For each divisor k (meaning each truck has k boxes),
        # the number of trucks is n // k. Compute that loading difference.
        for k in divisors:
            truck_count = n // k
            # If there's only one truck, difference is 0
            if truck_count == 1:
                diff = 0
            else:
                min_sum, max_sum = float('inf'), float('-inf')
                # Calculate each truck's total and track min & max
                for i in range(truck_count):
                    s = prefix[(i + 1) * k] - prefix[i * k]
                    if s < min_sum:
                        min_sum = s
                    if s > max_sum:
                        max_sum = s
                diff = max_sum - min_sum
            if diff > best_diff:
                best_diff = diff
        
        answers.append(str(best_diff))
    
    # Print all answers
    print("
".join(answers))

# Do not forget to call main() at the end.
if __name__ == "__main__":
    main()