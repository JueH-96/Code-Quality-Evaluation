def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    
    # Check for subarray sum S in the original array
    prefix = 0
    prefix_set = {0}
    found = False
    for num in A:
        prefix += num
        if (prefix - S) in prefix_set:
            found = True
            break
        prefix_set.add(prefix)
    if found:
        print("Yes")
        return
    
    total_sum = sum(A)
    if total_sum == 0:
        print("No")
        return
    
    rem = S % total_sum
    if rem == 0:
        if S >= total_sum:
            print("Yes")
        else:
            print("No")
    else:
        target = rem
        if target > S:
            print("No")
            return
        # Check if there's a subarray sum equal to target
        prefix = 0
        prefix_set = {0}
        found = False
        for num in A:
            prefix += num
            if (prefix - target) in prefix_set:
                found = True
                break
            prefix_set.add(prefix)
        print("Yes" if found else "No")

if __name__ == "__main__":
    main()