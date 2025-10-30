# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Count frequency of each number
    freq = defaultdict(int)
    for num in A:
        freq[num] += 1
    
    # Sort the unique numbers
    sorted_nums = sorted(freq.keys())
    
    total = 0
    
    # Precompute the prefix sum of frequencies
    prefix_freq = {}
    current_sum = 0
    for num in sorted_nums:
        current_sum += freq[num]
        prefix_freq[num] = current_sum
    
    # Iterate through each unique number
    for i in range(len(sorted_nums)):
        a = sorted_nums[i]
        # Calculate the contribution from pairs where a is the smaller number
        # Find all numbers b >= a
        # The floor of b / a is the same as b // a
        # So for each b >= a, the contribution is (b // a) * freq[a] * freq[b]
        # But when b == a, it's (a // a) * freq[a] * (freq[a] - 1) / 2
        # Because we are counting pairs (i,j) where i < j
        # So for b == a, the number of pairs is C(freq[a], 2) = freq[a] * (freq[a] - 1) / 2
        # And for b > a, the number of pairs is freq[a] * freq[b]
        
        # First handle the case where b == a
        if freq[a] >= 2:
            total += (a // a) * (freq[a] * (freq[a] - 1)) // 2
        
        # Now handle the case where b > a
        # We need to find all b > a and compute (b // a) * freq[a] * freq[b]
        # To optimize, we can iterate through all multiples of a and find the corresponding b
        # Since a can be up to 1e6, and N up to 2e5, this is manageable
        # For each multiple k*a, where k >= 2, we find the range of b in [k*a, (k+1)*a - 1]
        # and sum freq[b] for b in that range
        
        k = 2
        while k * a <= sorted_nums[-1]:
            lower = k * a
            upper = (k + 1) * a - 1
            # Find the indices in sorted_nums where b >= lower and b <= upper
            # Using binary search
            left = i
            right = len(sorted_nums) - 1
            # Find the first index >= lower
            low_index = -1
            while left <= right:
                mid = (left + right) // 2
                if sorted_nums[mid] >= lower:
                    low_index = mid
                    right = mid - 1
                else:
                    left = mid + 1
            if low_index == -1:
                k += 1
                continue
            # Find the last index <= upper
            left2 = low_index
            right2 = len(sorted_nums) - 1
            high_index = -1
            while left2 <= right2:
                mid = (left2 + right2) // 2
                if sorted_nums[mid] <= upper:
                    high_index = mid
                    left2 = mid + 1
                else:
                    right2 = mid - 1
            if high_index == -1:
                k += 1
                continue
            # Now, sum the frequencies from low_index to high_index
            # Using prefix_freq
            sum_freq = prefix_freq[sorted_nums[high_index]] - (prefix_freq[sorted_nums[low_index - 1]] if low_index > 0 else 0)
            total += k * freq[a] * sum_freq
            k += 1
    
    print(total)

if __name__ == "__main__":
    main()