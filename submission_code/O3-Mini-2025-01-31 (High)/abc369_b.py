def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # We separate the keys pressed by each hand.
    left_keys = []
    right_keys = []
    idx = 1
    for _ in range(N):
        key = int(data[idx])
        hand = data[idx + 1]
        idx += 2
        if hand == "L":
            left_keys.append(key)
        else:
            right_keys.append(key)
    
    # For each hand, it's optimal to start at the first pressed key (thus incurring no cost),
    # and then incur a cost equal to the absolute differences between consecutive keys
    # pressed using that hand.
    total_cost = 0
    if left_keys:
        prev = left_keys[0]
        for k in left_keys[1:]:
            total_cost += abs(k - prev)
            prev = k
            
    if right_keys:
        prev = right_keys[0]
        for k in right_keys[1:]:
            total_cost += abs(k - prev)
            prev = k
            
    sys.stdout.write(str(total_cost))

if __name__ == '__main__':
    main()