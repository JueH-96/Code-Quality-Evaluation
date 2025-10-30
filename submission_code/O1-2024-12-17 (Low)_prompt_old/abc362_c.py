def solve():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    pairs = []
    idx = 1
    for _ in range(N):
        L = int(input_data[idx]); R = int(input_data[idx+1])
        idx += 2
        pairs.append((L, R))
    
    # Compute the sum of all L_i and R_i
    S_min = sum(L for (L, _) in pairs)
    S_max = sum(R for (_, R) in pairs)
    
    # Check if 0 is in the range [S_min, S_max]
    if S_min > 0 or S_max < 0:
        print("No")
        return
    
    # We can try to construct a valid sequence X by starting with X_i = L_i
    X = [L for (L, _) in pairs]
    current_sum = S_min
    needed = -current_sum  # we want sum(X) to become 0
    
    # We will adjust each X_i upward as needed
    for i in range(N):
        if needed == 0:
            break
        L, R = pairs[i]
        can_increase = R - L  # how much we can increase X_i
        increase = min(needed, can_increase)
        X[i] += increase
        needed -= increase
    
    # After this, if needed != 0, then no solution
    if needed != 0:
        print("No")
    else:
        print("Yes")
        print(" ".join(map(str, X)))


def main():
    solve()

if __name__ == "__main__":
    main()