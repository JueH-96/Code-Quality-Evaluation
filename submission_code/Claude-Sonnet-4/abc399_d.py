def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Find positions of each number
    positions = {}
    for i in range(2 * n):
        if a[i] not in positions:
            positions[a[i]] = []
        positions[a[i]].append(i)
    
    count = 0
    
    # Check all pairs of couples
    for couple1 in range(1, n + 1):
        for couple2 in range(couple1 + 1, n + 1):
            pos1 = positions[couple1]
            pos2 = positions[couple2]
            
            # Check if neither couple is initially adjacent
            if abs(pos1[0] - pos1[1]) == 1 or abs(pos2[0] - pos2[1]) == 1:
                continue
            
            # Get all 4 positions
            all_positions = pos1 + pos2
            all_positions.sort()
            
            # Check if we can form two adjacent pairs
            # We need to check if there are two pairs of consecutive positions
            can_form = False
            
            # Try all ways to pair the 4 positions into 2 pairs
            # and check if both pairs can be consecutive
            positions_set = set(all_positions)
            
            # Check all possible ways to form two pairs of consecutive positions
            for i in range(len(all_positions)):
                for j in range(i + 1, len(all_positions)):
                    pos_a1, pos_a2 = all_positions[i], all_positions[j]
                    remaining = [p for k, p in enumerate(all_positions) if k != i and k != j]
                    pos_b1, pos_b2 = remaining[0], remaining[1]
                    
                    # Check if we can make both pairs adjacent
                    # For pair (pos_a1, pos_a2) to be adjacent, we need consecutive positions
                    # For pair (pos_b1, pos_b2) to be adjacent, we need consecutive positions
                    
                    # Check if pos_a1, pos_a2 can be made adjacent and pos_b1, pos_b2 can be made adjacent
                    # This means we need two pairs of consecutive positions among our 4 positions
                    
                    # Actually, let's think differently:
                    # We have 4 positions. We want to check if we can arrange them so that
                    # two specific pairs are consecutive.
                    
                    # The key insight: we can rearrange the values at these 4 positions freely.
                    # So we just need to check if among these 4 positions, there exist
                    # two pairs of consecutive positions.
                    
            # Let's check if there are at least 2 pairs of consecutive positions
            # among the 4 positions
            consecutive_pairs = 0
            used = set()
            
            for pos in all_positions:
                if pos in used:
                    continue
                if pos + 1 in positions_set and pos + 1 not in used:
                    consecutive_pairs += 1
                    used.add(pos)
                    used.add(pos + 1)
            
            if consecutive_pairs >= 2:
                can_form = True
            
            if can_form:
                count += 1
    
    return count

t = int(input())
for _ in range(t):
    print(solve())