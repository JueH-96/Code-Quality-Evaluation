import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total_sum = sum(A)
    result = []
    
    # To optimize, we can calculate the sum of all elements greater than A[i] by subtracting
    # the sum of all elements less than or equal to A[i] from the total sum.
    # We use a dictionary to count occurrences of each element.
    from collections import defaultdict
    count = defaultdict(int)
    for num in A:
        count[num] += 1
    
    # We need to know the sum of all elements <= A[i] for each A[i]
    # We can precompute the sum of elements <= any value using a prefix sum approach on counts.
    max_value = max(A)
    prefix_sum = [0] * (max_value + 1)
    
    for i in range(1, max_value + 1):
        prefix_sum[i] = prefix_sum[i - 1] + count[i] * i
    
    for a in A:
        sum_le_a = prefix_sum[a]
        sum_gt_a = total_sum - sum_le_a
        result.append(sum_gt_a)
    
    print(" ".join(map(str, result)))

main()