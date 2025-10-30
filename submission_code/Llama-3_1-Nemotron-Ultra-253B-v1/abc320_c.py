M = int(input())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

# Find all characters that are present in all three reels
common_chars = set(S1) & set(S2) & set(S3)
if not common_chars:
    print(-1)
    exit()

min_time = float('inf')

for c in common_chars:
    # Get positions where the character appears in each reel
    pos1 = [i for i, ch in enumerate(S1) if ch == c]
    pos2 = [i for i, ch in enumerate(S2) if ch == c]
    pos3 = [i for i, ch in enumerate(S3) if ch == c]
    
    # Earliest possible time for each reel
    t1 = min(pos1)
    t2 = min(pos2)
    t3 = min(pos3)
    
    current_max = max(t1, t2, t3)
    if current_max < min_time:
        min_time = current_max

print(min_time if min_time != float('inf') else -1)