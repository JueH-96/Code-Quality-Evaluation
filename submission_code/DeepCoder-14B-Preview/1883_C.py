import sys

def minimal_operations():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        k = int(input[ptr+1])
        ptr +=2
        a = list(map(int, input[ptr:ptr+n]))
        ptr +=n
        
        product_mod = 1
        for x in a:
            product_mod = (product_mod * (x % k)) % k
        
        if product_mod == 0:
            print(0)
            continue
        
        if k in [2, 3, 5]:
            min_steps = min( ( (k - (x % k)) % k for x in a ) )
            print(min_steps)
        elif k == 4:
            steps_single = min( ( (4 - (x % 4)) % 4 for x in a ) )
            steps_even = [x % 2 for x in a]
            steps_even_sorted = sorted(steps_even)
            if len(steps_even_sorted) >= 2:
                steps_pair = steps_even_sorted[0] + steps_even_sorted[1]
            else:
                steps_pair = float('inf')
            min_steps = min(steps_single, steps_pair)
            print(min_steps)
        else:
            # This case won't occur as per problem constraints
            pass

minimal_operations()