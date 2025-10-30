import itertools

def find_kth_321_like(K):
    numbers = []
    for d in range(1, 11):  # d is the number of digits
        if d == 1:
            for i in range(1, 10):
                numbers.append(i)
        else:
            # Generate all combinations of d distinct digits from 9 down to 0
            for comb in itertools.combinations(range(9, -1, -1), d):
                # Convert the combination to a number
                num = 0
                for c in comb:
                    num = num * 10 + c
                numbers.append(num)
    # Sort all generated numbers
    numbers.sort()
    # Return the K-th smallest number
    return numbers[K-1]

# Read input
K = int(input())
# Find and print the result
print(find_kth_321_like(K))