import itertools

def find_kth_321_like_number(K):
    # Precompute all possible 321-like numbers for each digit length and sort them
    d_lists = []
    for d in range(1, 11):
        if d == 1:
            # Single-digit numbers are 1-9
            numbers = list(range(1, 10))
        else:
            # Generate all combinations of d distinct digits from 0-9
            numbers = []
            for combo in itertools.combinations(range(0, 10), d):
                # Sort the combination in reverse to get decreasing order
                sorted_combo = sorted(combo, reverse=True)
                # Convert the combination to a number
                num = 0
                for digit in sorted_combo:
                    num = num * 10 + digit
                numbers.append(num)
            # Sort the numbers for this digit length
            numbers.sort()
        d_lists.append(numbers)
    
    # Find the target digit length
    sum_counts = 0
    for d in range(1, 11):
        if d == 1:
            cnt = 9
        else:
            cnt = len(d_lists[d-1])
        if sum_counts + cnt >= K:
            target_d = d
            index = K - sum_counts - 1
            return d_lists[target_d - 1][index]
        else:
            sum_counts += cnt

# Read input and output the result
K = int(input())
print(find_kth_321_like_number(K))