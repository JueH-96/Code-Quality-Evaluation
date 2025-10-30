N, K = map(int, input().split())
A = list(map(int, input().split()))

def get_valid_cuts(min_weight):
    valid_cuts = set()
    
    # Try all possible starting positions
    for start in range(N):
        cuts = []
        current_weight = 0
        people = 0
        
        for i in range(N):
            current_weight += A[(start + i) % N]
            
            # If we have enough weight and haven't assigned to K-1 people yet
            if current_weight >= min_weight and people < K - 1:
                cuts.append((start + i) % N)
                people += 1
                current_weight = 0
        
        # Check if the last person also gets at least min_weight
        if people == K - 1 and current_weight >= min_weight:
            valid_cuts.add(tuple(cuts))
    
    return valid_cuts

def can_achieve_min_weight(min_weight):
    return len(get_valid_cuts(min_weight)) > 0

# Binary search for the maximum achievable minimum weight
left, right = 1, sum(A)
max_min_weight = 0

while left <= right:
    mid = (left + right) // 2
    if can_achieve_min_weight(mid):
        max_min_weight = mid
        left = mid + 1
    else:
        right = mid - 1

# Get all valid cuts for the optimal weight
all_valid_cuts = get_valid_cuts(max_min_weight)

# Find all cut lines that are used in any optimal division
used_cut_lines = set()
for cuts in all_valid_cuts:
    for cut in cuts:
        used_cut_lines.add(cut)

never_cut_count = N - len(used_cut_lines)

print(max_min_weight, never_cut_count)