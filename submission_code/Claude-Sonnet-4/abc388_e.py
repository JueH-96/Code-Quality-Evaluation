n = int(input())
a = list(map(int, input().split()))

left = 0  # pointer for smaller mochi (top)
right = 0  # pointer for larger mochi (bottom)
pairs = 0

while left < n:
    # Find the first mochi that the current left mochi can be placed on
    while right < n and (right <= left or a[left] > a[right] // 2):
        right += 1
    
    if right < n:
        pairs += 1
        right += 1  # This mochi is used as base
    
    left += 1

print(pairs)