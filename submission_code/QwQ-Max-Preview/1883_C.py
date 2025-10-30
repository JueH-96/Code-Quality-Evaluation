import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n, k = int(input[idx]), int(input[idx+1])
        idx +=2
        a = list(map(int, input[idx:idx+n]))
        idx +=n
        product_mod = 1
        for num in a:
            product_mod = (product_mod * (num % k)) % k
        if product_mod == 0:
            print(0)
            continue
        if k == 2:
            print(1)
        elif k == 3:
            min_step = min( (3 - (num % 3)) %3 for num in a )
            print(min_step)
        elif k ==4:
            sum_exp =0
            for num in a:
                exp =0
                x = num
                while x %2 ==0 and x !=0:
                    exp +=1
                    x = x//2
                sum_exp += exp
            if sum_exp >=2:
                print(0)
                continue
            needed = 2 - sum_exp
            if needed ==1:
                steps = [ (2 - (num%2))%2 for num in a ]
                print( min(steps) )
            else:
                steps_to_1 = [ (2 - (num%2))%2 for num in a ]
                steps_to_2 = []
                for num in a:
                    mod = num %4
                    if mod ==0:
                        steps_to_2.append(0)
                    else:
                        steps_to_2.append( (4 - mod) %4 )
                option1 = min(steps_to_2)
                sorted_1 = sorted(steps_to_1)
                option2 = sorted_1[0] + sorted_1[1]
                print( min(option1, option2) )
        elif k ==5:
            min_step = min( (5 - (num %5)) %5 for num in a )
            print(min_step)
        
if __name__ == "__main__":
    main()