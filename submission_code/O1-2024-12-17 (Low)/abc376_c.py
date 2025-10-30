def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:1+N]))
    B = list(map(int, input_data[1+N:1+N+(N-1)]))
    
    # Quick impossibility check:
    # We must use all N-1 existing boxes plus the new one. If any existing
    # box is smaller than every toy (i.e. smaller than min(A)), it's useless,
    # so no arrangement is possible.
    minA = min(A)
    if any(b < minA for b in B):
        print(-1)
        return
    
    # Sort toys and boxes
    A.sort()
    B.sort()
    
    # Greedy check function:
    # Checks if we can store all toys in B plus one new box of size x.
    def can_store(x):
        i = 0  # index for toys
        j = 0  # index for existing boxes
        used_new_box = False
        while i < N:
            # skip boxes that are too small for A[i]
            while j < N - 1 and B[j] < A[i]:
                j += 1
            if j < N - 1 and B[j] >= A[i]:
                # use this existing box
                j += 1
                i += 1
            elif not used_new_box and x >= A[i]:
                # use the new purchased box
                used_new_box = True
                i += 1
            else:
                return False
        return True
    
    # We only need to search in [0..10^9], because A_i, B_i <= 10^9
    # If even x=10^9 doesn't work, there's no solution.
    left, right = 0, 10**9
    answer = -1
    while left <= right:
        mid = (left + right) // 2
        if can_store(mid):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(answer if answer != -1 else -1)

# Call main() at the end
if __name__ == "__main__":
    main()