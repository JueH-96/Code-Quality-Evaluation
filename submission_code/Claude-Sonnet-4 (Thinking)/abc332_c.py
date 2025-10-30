N, M = map(int, input().split())
S = input().strip()

# Split the schedule into segments separated by '0' (washing days)
segments = []
current_segment = ""

for c in S:
    if c == '0':
        if current_segment:
            segments.append(current_segment)
            current_segment = ""
    else:
        current_segment += c

if current_segment:
    segments.append(current_segment)

max_logo_needed = 0

for segment in segments:
    count_1 = segment.count('1')
    count_2 = segment.count('2')
    
    logo_needed_for_segment = count_2 + max(0, count_1 - M)
    max_logo_needed = max(max_logo_needed, logo_needed_for_segment)

print(max_logo_needed)