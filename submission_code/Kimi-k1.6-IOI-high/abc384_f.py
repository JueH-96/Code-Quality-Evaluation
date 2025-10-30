import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    sum_m = [0] * 27  # sum_m[0] to sum_m[26], sum_m[26] is 0
    
    for m in range(26):
        d = 1 << m
        counts = defaultdict(int)
        sums = defaultdict(int)
        current_sum = 0
        for num in A:
            rem_needed = (-num) % d
            cnt = counts.get(rem_needed, 0)
            sum_val = sums.get(rem_needed, 0)
            current_sum += num * cnt + sum_val
            # Check if (j,j) pair is valid
            if (2 * num) % d == 0:
                current_sum += 2 * num
            # Update counts and sums for current num
            rem = num % d
            counts[rem] += 1
            sums[rem] += num
        sum_m[m] = current_sum
    
    total = 0
    for m in range(26):
        numerator = sum_m[m] - sum_m[m + 1]
        denominator = 1 << m
        total += numerator // denominator
    
    print(total)

if __name__ == '__main__':
    main()