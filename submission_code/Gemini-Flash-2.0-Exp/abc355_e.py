def solve():
    n, l, r = map(int, input().split())
    
    total_sum = 0
    for i in range(l, r + 1):
        power = 0
        while True:
            length = 2**power
            start = (i // length) * length
            end = start + length - 1
            
            if start >= l and end <= r:
                print("? {} {}".format(power, i // length))
                
                response = int(input())
                if response == -1:
                    return
                
                total_sum = (total_sum + response) % 100
                
                for j in range(start, end + 1):
                    if l <= j <= r:
                        if j == i:
                            break
                
                i = end
                break
            else:
                power += 1
                if power > n:
                    break
    
    print("! {}".format(total_sum))

solve()