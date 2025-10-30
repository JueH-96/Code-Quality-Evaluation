# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    if k == 2:
        # Need at least one even element
        if any(x % 2 == 0 for x in a):
            print(0)
        else:
            print(1)
    
    elif k == 3:
        # Need at least one element divisible by 3
        min_ops = min((3 - (x % 3)) % 3 for x in a)
        print(min_ops)
    
    elif k == 5:
        # Need at least one element divisible by 5
        min_ops = min((5 - (x % 5)) % 5 for x in a)
        print(min_ops)
    
    elif k == 4:
        # Case 1: Make one element divisible by 4
        min_ops1 = min((4 - (x % 4)) % 4 for x in a)
        
        # Case 2: Make two elements divisible by 2
        even_count = sum(1 for x in a if x % 2 == 0)
        if even_count >= 2:
            min_ops2 = 0
        elif even_count == 1:
            min_ops2 = 1  # Make one more element even
        else:
            min_ops2 = 2  # Make two elements even
        
        print(min(min_ops1, min_ops2))