import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n, k = int(input[ptr]), int(input[ptr+1])
        ptr +=2
        a = list(map(int, input[ptr:ptr+n]))
        ptr +=n
        
        if k in [2,3,5]:
            found = False
            for num in a:
                if num % k == 0:
                    found = True
                    break
            if found:
                print(0)
                continue
            min_step = float('inf')
            for num in a:
                mod = num % k
                step = (k - mod) % k
                if step < min_step:
                    min_step = step
            print(min_step)
        else:
            # k ==4
            sum_exponents = 0
            exponents = []
            for num in a:
                tmp = num
                cnt =0
                while tmp %2 ==0:
                    cnt +=1
                    tmp //=2
                exponents.append(cnt)
                sum_exponents += cnt
            if sum_exponents >=2:
                print(0)
                continue
            if sum_exponents ==0:
                option1 = float('inf')
                for num in a:
                    step = (4 - (num %4 )) %4
                    if step < option1:
                        option1 = step
                ans = min(option1, 2)
                print(ans)
            else: # sum_exponents ==1
                min_step = float('inf')
                for i in range(n):
                    num = a[i]
                    exp = exponents[i]
                    if exp ==0:
                        curr_step =1
                    else:
                        # exp ==1
                        curr_step = (4 - (num %4 )) %4
                    if curr_step < min_step:
                        min_step = curr_step
                print(min_step)
                    
                
            
if __name__ == "__main__":
    main()