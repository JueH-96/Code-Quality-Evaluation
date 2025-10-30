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
        
        if k in {2,3,5}:
            has_div = any(x % k == 0 for x in a)
            if has_div:
                print(0)
                continue
            min_steps = float('inf')
            for x in a:
                rem = x % k
                steps = (k - rem) % k
                if steps < min_steps:
                    min_steps = steps
            print(min_steps)
        elif k ==4:
            steps1 = []
            steps2 = []
            for x in a:
                rem1 = x % 2
                steps1_val = (2 - rem1) % 2
                steps1.append(steps1_val)
                
                rem2 = x %4
                steps2_val = (4 - rem2) %4
                steps2.append(steps2_val)
            
            min_steps2 = min(steps2)
            sum_steps1 = float('inf')
            if len(steps1) >=2:
                steps1_sorted = sorted(steps1)
                sum_steps1 = steps1_sorted[0] + steps1_sorted[1]
            ans = min(min_steps2, sum_steps1)
            print(ans)
        else:
            print(0)  # This case should not occur as per problem constraints

if __name__ == "__main__":
    main()