def count_pieces(pos, N):
    # Count number of pieces in each position
    count = [0] * N
    for p in pos:
        if 0 <= p < N:
            count[p] += 1
    return count

def can_reach_target(curr_pieces, target, N):
    # Check if current configuration matches target
    for i in range(N):
        if (curr_pieces[i] > 0) != (target[i] == '1'):
            return False
    return True

def solve_case():
    N = int(input())
    A = input().strip()
    B = input().strip()
    
    # Get initial piece positions
    pieces = []
    for i in range(N):
        if A[i] == '1':
            pieces.append(i)
    
    # If number of pieces doesn't match target, impossible
    if sum(1 for c in A if c == '1') != sum(1 for c in B if c == '1'):
        return -1
        
    # Try each possible center position
    min_ops = float('inf')
    for center in range(N):
        curr_pieces = pieces.copy()
        ops = 0
        
        # Keep moving pieces until they stop moving or exceed reasonable limit
        while ops < N * 2:
            # Count current pieces in each position
            counts = count_pieces(curr_pieces, N)
            
            # Check if we reached target configuration
            if can_reach_target(counts, B, N):
                min_ops = min(min_ops, ops)
                break
                
            # Move all pieces one step towards center
            moved = False
            for i in range(len(curr_pieces)):
                if curr_pieces[i] < center:
                    curr_pieces[i] += 1
                    moved = True
                elif curr_pieces[i] > center:
                    curr_pieces[i] -= 1
                    moved = True
            
            if not moved:
                break
                
            ops += 1
    
    return min_ops if min_ops != float('inf') else -1

# Read number of test cases
T = int(input())

# Process each test case
for _ in range(T):
    result = solve_case()
    print(result)