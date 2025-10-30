def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Find positions of each couple
    positions = {}
    for i in range(2 * n):
        if a[i] not in positions:
            positions[a[i]] = []
        positions[a[i]].append(i)
    
    count = 0
    
    # Check all pairs (a, b) where a < b
    for couple_a in range(1, n + 1):
        for couple_b in range(couple_a + 1, n + 1):
            pos_a = positions[couple_a]
            pos_b = positions[couple_b]
            
            # Check if neither couple is initially adjacent
            if abs(pos_a[0] - pos_a[1]) == 1:
                continue
            if abs(pos_b[0] - pos_b[1]) == 1:
                continue
            
            # Check if we can make both couples adjacent by swapping
            # We need to check all possible swap combinations
            found = False
            
            # Try swapping pos_a[0] with pos_b[0], pos_a[1] with pos_b[1]
            if check_swap(pos_a[0], pos_b[0], pos_a[1], pos_b[1]):
                found = True
            
            # Try swapping pos_a[0] with pos_b[1], pos_a[1] with pos_b[0]
            if not found and check_swap(pos_a[0], pos_b[1], pos_a[1], pos_b[0]):
                found = True
            
            if found:
                count += 1
    
    return count

def check_swap(p1, p2, p3, p4):
    # After swapping p1 with p2 and p3 with p4
    # Check if both couples become adjacent
    
    # Check if the positions form valid adjacent pairs
    positions = sorted([p1, p2, p3, p4])
    
    # For both couples to be adjacent, the 4 positions must form 2 consecutive pairs
    # This means they should be in positions like (0,1,2,3) or (2,3,4,5) etc.
    # Or they could be in two separate consecutive pairs like (0,1) and (4,5)
    
    # Case 1: All 4 positions are consecutive
    if positions[3] - positions[0] == 3:
        # Check if they form proper pairs
        if (positions[0] % 2 == 0 and positions[1] == positions[0] + 1 and 
            positions[2] % 2 == 0 and positions[3] == positions[2] + 1):
            return True
    
    # Case 2: Two separate consecutive pairs
    if (positions[1] == positions[0] + 1 and positions[3] == positions[2] + 1):
        # Check if both pairs start at even positions
        if positions[0] % 2 == 0 and positions[2] % 2 == 0:
            # Check if the swap actually makes both couples adjacent
            # p1 goes to p2's position, p2 goes to p1's position
            # p3 goes to p4's position, p4 goes to p3's position
            new_positions = {}
            if p1 in [positions[0], positions[1]] and p2 in [positions[0], positions[1]]:
                if p3 in [positions[2], positions[3]] and p4 in [positions[2], positions[3]]:
                    return True
            if p1 in [positions[0], positions[1]] and p2 in [positions[2], positions[3]]:
                if p3 in [positions[0], positions[1]] and p4 in [positions[2], positions[3]]:
                    return True
            if p1 in [positions[2], positions[3]] and p2 in [positions[0], positions[1]]:
                if p3 in [positions[2], positions[3]] and p4 in [positions[0], positions[1]]:
                    return True
    
    return False

# Read number of test cases
t = int(input())
for _ in range(t):
    result = solve()
    print(result)