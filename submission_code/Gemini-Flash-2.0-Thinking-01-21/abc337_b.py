def solve():
    s = input()
    n = len(s)
    index = 0
    
    # Consume 'A's
    while index < n and s[index] == 'A':
        index += 1
        
    # Consume 'B's
    while index < n and s[index] == 'B':
        index += 1
        
    # Consume 'C's
    while index < n and s[index] == 'C':
        index += 1
        
    if index == n:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()