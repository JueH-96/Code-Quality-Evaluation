# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    if k == 2:
        # Need at least one even number
        min_ops = float('inf')
        for x in a:
            if x % 2 == 0:
                return 0
            min_ops = min(min_ops, 1)  # Any odd number needs 1 operation
        return min_ops
    
    elif k == 3:
        # Need at least one number divisible by 3
        min_ops = float('inf')
        for x in a:
            if x % 3 == 0:
                return 0
            # Operations needed to make x divisible by 3
            ops = 3 - (x % 3)
            min_ops = min(min_ops, ops)
        return min_ops
    
    elif k == 4:
        # Need at least two factors of 2
        # Option 1: Make one number divisible by 4
        min_ops1 = float('inf')
        for x in a:
            if x % 4 == 0:
                min_ops1 = 0
                break
            # Operations needed to make x divisible by 4
            ops = 4 - (x % 4)
            min_ops1 = min(min_ops1, ops)
        
        # Option 2: Make two numbers even
        even_count = sum(1 for x in a if x % 2 == 0)
        if even_count >= 2:
            min_ops2 = 0
        elif even_count == 1:
            min_ops2 = 1  # Need to make one more number even
        else:
            min_ops2 = 2  # Need to make two numbers even
        
        return min(min_ops1, min_ops2)
    
    elif k == 5:
        # Need at least one number divisible by 5
        min_ops = float('inf')
        for x in a:
            if x % 5 == 0:
                return 0
            # Operations needed to make x divisible by 5
            ops = 5 - (x % 5)
            min_ops = min(min_ops, ops)
        return min_ops

t = int(input())
for _ in range(t):
    print(solve())