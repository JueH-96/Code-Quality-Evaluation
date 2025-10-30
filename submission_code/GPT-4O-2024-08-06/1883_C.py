def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        n = int(data[index])
        k = int(data[index + 1])
        index += 2
        a = list(map(int, data[index:index + n]))
        index += n
        
        if k == 2:
            # Check for even numbers
            has_even = any(x % 2 == 0 for x in a)
            if has_even:
                results.append(0)
            else:
                results.append(1)
        
        elif k == 3:
            # Check for numbers divisible by 3
            has_div3 = any(x % 3 == 0 for x in a)
            if has_div3:
                results.append(0)
            else:
                results.append(1)
        
        elif k == 4:
            # Check for numbers divisible by 4 or at least two evens
            count_even = sum(1 for x in a if x % 2 == 0)
            has_div4 = any(x % 4 == 0 for x in a)
            if has_div4 or count_even >= 2:
                results.append(0)
            elif count_even == 1:
                results.append(1)
            else:
                results.append(2)
        
        elif k == 5:
            # Check for numbers divisible by 5
            has_div5 = any(x % 5 == 0 for x in a)
            if has_div5:
                results.append(0)
            else:
                results.append(1)
    
    sys.stdout.write("
".join(map(str, results)) + "
")