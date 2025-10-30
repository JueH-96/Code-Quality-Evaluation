# Read input
N, D = map(int, input().split())
S = input()

# Convert string to list for easier manipulation
boxes = list(S)

# Simulate D days
for _ in range(D):
    # Find rightmost cookie
    rightmost_cookie = -1
    for i in range(N-1, -1, -1):
        if boxes[i] == '@':
            rightmost_cookie = i
            break
    
    # Remove the rightmost cookie
    if rightmost_cookie != -1:
        boxes[rightmost_cookie] = '.'

# Print final state
print(''.join(boxes))