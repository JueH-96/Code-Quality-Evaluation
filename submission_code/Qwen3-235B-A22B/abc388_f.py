def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = int(input[ptr])
    ptr += 1
    B = int(input[ptr])
    ptr += 1

    forbidden = []
    for _ in range(M):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        forbidden.append((L, R))
    
    # Generate allowed intervals
    allowed = []
    prev_end = 0
    for L, R in forbidden:
        start = prev_end + 1
        end = L - 1
        if start <= end:
            allowed.append((start, end))
        prev_end = R
    # Handle the last allowed interval
    start = prev_end + 1
    end = N
    if start <= end:
        allowed.append((start, end))
    
    # Edge case: allowed is empty (unlikely as forbidden can't cover N)
    if not allowed:
        print("No")
        return
    
    # Initialize reachable range
    low = 1
    high = 1
    reached = False

    for s, e in allowed:
        if s > high + B:
            break
        # Compute new_min and new_max
        new_min = max(s, low + A)
        new_max = min(e, high + B)
        if new_min <= new_max:
            low = new_min
            if e > high:
                high = e
            if high >= N:
                reached = True
                break
    
    if reached or high >= N:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()