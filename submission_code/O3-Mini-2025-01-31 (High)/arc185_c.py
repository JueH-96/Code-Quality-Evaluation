def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Create a list of pairs (value, original_index)
    nums = [(A[i], i+1) for i in range(N)]
    # Sort by the value
    nums.sort(key=lambda x: x[0])
    
    # Use three pointer approach: iterate i and then two-sum on the rest of the array.
    for i in range(N - 2):
        target = X - nums[i][0]
        left = i + 1
        right = N - 1
        while left < right:
            current_sum = nums[left][0] + nums[right][0]
            if current_sum == target:
                # Found a valid triple; retrieve original indices and sort them (i < j < k)
                res = sorted([nums[i][1], nums[left][1], nums[right][1]])
                sys.stdout.write(f"{res[0]} {res[1]} {res[2]}")
                return
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    # If no triple is found, output -1.
    sys.stdout.write("-1")

if __name__ == '__main__':
    main()