# Read input
N, M = map(int, input().split())
S = input().strip()

# Split the schedule into segments separated by '0'
segments = S.split('0')

max_logo_needed = 0

for seg in segments:
    cnt1 = seg.count('1')
    cnt2 = seg.count('2')
    # Calculate the required logo T-shirts for this segment
    required = cnt2 + max(0, cnt1 - M)
    if required > max_logo_needed:
        max_logo_needed = required

# Output the result
print(max_logo_needed)