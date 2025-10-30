import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n, k = int(input[ptr]), int(input[ptr+1])
        ptr += 2
        a = list(map(int, input[ptr:ptr+n]))
        ptr += n
        
        if k == 2:
            if any(x % 2 == 0 for x in a):
                print(0)
            else:
                print(1)
        elif k == 3:
            has = False
            for x in a:
                if x % 3 == 0:
                    has = True
                    break
            if has:
                print(0)
            else:
                min_steps = float('inf')
                for x in a:
                    rem = x % 3
                    steps = 3 - rem if rem != 0 else 0
                    if steps < min_steps:
                        min_steps = steps
                print(min_steps)
        elif k == 4:
            sum_2 = 0
            for num in a:
                cnt = 0
                tmp = num
                while tmp % 2 == 0:
                    cnt += 1
                    tmp //= 2
                sum_2 += cnt
            if sum_2 >= 2:
                print(0)
            elif sum_2 == 0:
                min_step_4_0 = float('inf')
                for num in a:
                    rem = num % 4
                    steps = 4 - rem if rem != 0 else 0
                    if steps < min_step_4_0:
                        min_step_4_0 = steps
                print(min(2, min_step_4_0))
            else:
                print(1)
        elif k == 5:
            if any(x % 5 == 0 for x in a):
                print(0)
            else:
                min_steps = float('inf')
                for x in a:
                    rem = x % 5
                    steps = 5 - rem if rem != 0 else 0
                    if steps < min_steps:
                        min_steps = steps
                print(min_steps)

if __name__ == "__main__":
    main()