def solve():
    import sys
    s = list(map(int, sys.stdin.read().split()))
    
    # Condition 1: Non-decreasing order
    if any(s[i] > s[i+1] for i in range(7)):
        print("No")
        return
    
    # Condition 2: All between 100 and 675 inclusive
    if any(x < 100 or x > 675 for x in s):
        print("No")
        return
    
    # Condition 3: All multiples of 25
    if any(x % 25 != 0 for x in s):
        print("No")
        return
    
    # If all conditions are met
    print("Yes")

# Call solve
solve()