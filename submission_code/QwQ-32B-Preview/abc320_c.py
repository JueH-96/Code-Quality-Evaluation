def main():
    import sys

    M = int(sys.stdin.readline())
    S1 = sys.stdin.readline().strip()
    S2 = sys.stdin.readline().strip()
    S3 = sys.stdin.readline().strip()

    # Precompute allowed times for each reel and each digit
    reels = [S1, S2, S3]
    allowed_times = []
    for reel in reels:
        reel_allowed = {}
        for j in range(1, M + 1):
            d = reel[j - 1]
            mod_val = (j - 1) % M
            if d not in reel_allowed:
                reel_allowed[d] = []
            reel_allowed[d].append(mod_val)
        allowed_times.append(reel_allowed)

    # Function to check if a given t is valid for some digit d
    def is_valid(t):
        for d in '0123456789':
            found = True
            for i in range(3):
                if d not in allowed_times[i]:
                    found = False
                    break
                # Check if there exists allowed <= t for this reel and digit
                reel_allowed = allowed_times[i][d]
                if not any(allowed <= t for allowed in reel_allowed):
                    found = False
                    break
            if found:
                return True
        return False

    # Binary search for the minimal t
    left = 0
    right = M * M  # Upper bound is M*M, should be sufficient
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if is_valid(mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    print(result)

if __name__ == "__main__":
    main()