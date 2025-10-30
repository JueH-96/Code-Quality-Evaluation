# YOUR CODE HERE
t = int(input())
for _ in range(t):
    s = input().strip()
    target = "abc"
    
    # Count positions that differ from target
    diff_count = 0
    for i in range(3):
        if s[i] != target[i]:
            diff_count += 1
    
    # If 0 or 2 positions differ, we can make it "abc" with at most one swap
    if diff_count == 0 or diff_count == 2:
        print("YES")
    else:
        print("NO")