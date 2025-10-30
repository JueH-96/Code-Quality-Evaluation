def can_find_subsequence(N, S, A):
    # Calculate the prefix sums for the first N elements
    prefix_sums = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + A[i - 1]

    # Check if we can find a contiguous subsequence that sums to S
    # We will use a set to store the prefix sums we have seen
    seen_sums = set()
    
    # We will iterate through the prefix sums
    for i in range(N + 1):
        # Check if we can form the desired sum S
        if (prefix_sums[i] - S) in seen_sums:
            return "Yes"
        
        # Add the current prefix sum to the seen sums
        seen_sums.add(prefix_sums[i])
        
        # Since the sequence is periodic, we can check for sums that wrap around
        # We can check the prefix sums with an offset of k * total_sum
        total_sum = prefix_sums[N]
        if total_sum > 0:  # Only check this if the total sum is positive
            # We need to check if we can find a prefix sum that matches
            # (prefix_sums[i] - S) + k * total_sum for some k >= 0
            # Rearranging gives us: prefix_sums[i] - S - k * total_sum = 0
            # This means we need to check if we can find a prefix sum
            # that is equal to (prefix_sums[i] - S) + k * total_sum
            for k in range(1, (S // total_sum) + 2):  # Check for k = 1 to k = (S // total_sum) + 1
                if (prefix_sums[i] - S) + k * total_sum in seen_sums:
                    return "Yes"

    return "No"

import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    S = int(data[1])
    A = list(map(int, data[2:2 + N]))
    
    result = can_find_subsequence(N, S, A)
    print(result)

if __name__ == "__main__":
    main()