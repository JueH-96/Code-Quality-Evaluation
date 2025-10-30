import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N

    # Compute prefix sums
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    T = P[N]

    # Check if S is less than the minimal element
    min_A = min(A)
    if S < min_A:
        print("No")
        return

    # Case 1: Check for any subarray within one period
    found = False
    prev = {0}
    for j in range(1, N+1):
        current_P = P[j]
        if (current_P - S) in prev:
            found = True
            break
        prev.add(current_P)
    if found:
        print("Yes")
        return

    # Proceed to case 2 if S >= T
    if S < T:
        print("No")
        return

    r = S % T

    # Initialize mod_dict with P[0]
    mod_dict = {}
    mod0 = P[0] % T
    mod_dict[mod0] = [P[0]]

    for j in range(1, N+1):
        current_P = P[j]
        current_mod = current_P % T
        target_mod = (current_mod - r) % T

        if target_mod in mod_dict:
            list_Pi = mod_dict[target_mod]
            threshold = current_P - (S - T)
            idx_bisect = bisect.bisect_left(list_Pi, threshold)
            if idx_bisect < len(list_Pi):
                print("Yes")
                return

        # Add current_P to mod_dict's current_mod list
        if current_mod in mod_dict:
            mod_dict[current_mod].append(current_P)
        else:
            mod_dict[current_mod] = [current_P]

    # If no solution found
    print("No")

if __name__ == "__main__":
    main()