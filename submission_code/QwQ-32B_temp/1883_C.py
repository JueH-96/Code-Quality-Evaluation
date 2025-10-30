import sys

def main():
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
        if k == 2:
            has_even = any(x % 2 == 0 for x in a)
            print(0 if has_even else 1)
        elif k == 3:
            has_div3 = any(x % 3 == 0 for x in a)
            if has_div3:
                print(0)
            else:
                min_step = min((3 - (x % 3)) % 3 for x in a)
                print(min_step)
        elif k ==4:
            has_div4 = False
            even_count =0
            for x in a:
                if x %4 ==0:
                    has_div4 = True
                if x %2 ==0:
                    even_count +=1
            if has_div4 or even_count >=2:
                print(0)
            else:
                steps1 = min( (4 - (x%4))%4 for x in a )
                steps2 = (2 - even_count)
                print(min(steps1, steps2))
        elif k ==5:
            has_div5 = any(x %5 ==0 for x in a)
            if has_div5:
                print(0)
            else:
                min_step = min( (5 - (x%5))%5 for x in a )
                print(min_step)

if __name__ == "__main__":
    main()