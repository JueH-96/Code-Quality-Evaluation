import sys

def count_factors(num, factor):
    cnt = 0
    while num % factor == 0 and num != 0:
        cnt += 1
        num = num // factor
    return cnt

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
        
        if k == 1:
            print(0)
            continue
        
        if k in (2, 3, 5):
            total = 0
            for num in a:
                total += count_factors(num, k)
                if total >= 1:
                    break
            if total >= 1:
                print(0)
            else:
                min_steps = float('inf')
                for num in a:
                    rem = num % k
                    if rem == 0:
                        steps = 0
                    else:
                        steps = k - rem
                    if steps < min_steps:
                        min_steps = steps
                print(min_steps)
        elif k == 4:
            current_2 = 0
            for num in a:
                current_2 += count_factors(num, 2)
            if current_2 >= 2:
                print(0)
            else:
                deficit = 2 - current_2
                best_1 = []
                best_2 = []
                for num in a:
                    original_2 = count_factors(num, 2)
                    b1 = float('inf')
                    b2 = float('inf')
                    for x in range(0, 21):
                        new_num = num + x
                        new_2 = count_factors(new_num, 2)
                        gain = new_2 - original_2
                        if gain >= 1 and x < b1:
                            b1 = x
                        if gain >= 2 and x < b2:
                            b2 = x
                    best_1.append(b1)
                    best_2.append(b2)
                if deficit == 1:
                    print(min(best_1))
                else:
                    min_b2 = min(best_2)
                    sorted_b1 = sorted(best_1)
                    sum_b1 = sorted_b1[0] + sorted_b1[1]
                    print(min(min_b2, sum_b1))
        else:
            print(0)

if __name__ == "__main__":
    main()