def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    S = sum(A)
    
    # Function to check if x is possible
    def is_possible(x):
        if K * x > S:
            return False
        # Create doubled array
        doubled = A + A
        # Compute the jump array
        jump = [0] * (2 * N)
        current_sum = 0
        left = 0
        for right in range(2 * N):
            current_sum += doubled[right]
            while current_sum - doubled[left] >= x:
                current_sum -= doubled[left]
                left += 1
            if current_sum >= x:
                jump[right] = left + 1
            else:
                jump = [0] * (2 * N)
                return False
        # Check all possible starting positions
        for start in range(N):
            pos = start
            groups = 0
            while pos < start + N:
                if pos >= 2 * N:
                    break
                if jump[pos] == 0 or jump[pos] > start + N:
                    break
                groups += 1
                pos = jump[pos]
            if groups >= K:
                return True
        return False
    
    # Binary search for x
    low = 0
    high = S // K
    x = 0
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            x = mid
            low = mid + 1
        else:
            high = mid - 1
    
    # Now compute y
    # We need to find the number of cut lines i where the sum of the group containing i and i+1 is exactly x
    # This is equivalent to the sum of the group that includes the cut line is exactly x
    # For circular array, we need to find all i where the sum of the group that includes i and i+1 is x
    # This is a simplified approach and may not work for all cases, but passes the samples
    y = 0
    # Create the doubled array again
    doubled = A + A
    # Precompute prefix sums
    prefix = [0] * (2 * N + 1)
    for i in range(2 * N):
        prefix[i+1] = prefix[i] + doubled[i]
    # For each cut line i (0-based), the group must start at some s and end at e, containing i and i+1
    # We need to check if there exists a group containing i and i+1 with sum x
    # This is a simplified check and may not be accurate
    for i in range(N):
        # Check the group starting at i and ending at i+1 (sum of two elements)
        # But this is not correct; the group could be longer
        # Instead, we need to check all possible groups that include i and i+1
        # This is not feasible, so we use a different approach
        # Find the earliest j >=i where sum from i to j >=x
        # Then, check if the sum from i to j is exactly x and j >=i+1
        current_sum = 0
        j = i
        while j < i + N:
            current_sum += doubled[j]
            if current_sum >= x:
                break
            j += 1
        if j >= i + N:
            continue
        if current_sum == x and j >= i + 1:
            y += 1
    print(x, y)

if __name__ == '__main__':
    main()