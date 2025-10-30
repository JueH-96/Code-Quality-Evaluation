import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    factors_of_2 = [0] * 11
    for x in range(1, 11):
        count = 0
        temp = x
        while temp % 2 == 0:
            count += 1
            temp //= 2
        factors_of_2[x] = count
    
    t = int(data[0])
    ptr = 1
    for _ in range(t):
        if ptr >= len(data):
            break
        n_k = data[ptr].strip()
        while n_k == '':
            ptr += 1
            n_k = data[ptr].strip()
        n, k = map(int, n_k.split())
        ptr += 1
        a_line = data[ptr].strip()
        while a_line == '':
            ptr += 1
            a_line = data[ptr].strip()
        a = list(map(int, a_line.split()))
        ptr += 1
        
        if k == 2:
            has_even = any(x % 2 == 0 for x in a)
            if has_even:
                print(0)
            else:
                print(1)
        elif k == 3:
            has_div3 = any(x % 3 == 0 for x in a)
            if has_div3:
                print(0)
            else:
                ops = [3 - x % 3 for x in a if x % 3 != 0]
                print(min(ops))
        elif k == 4:
            total_factors = sum(factors_of_2[x] for x in a)
            if total_factors >= 2:
                print(0)
            elif total_factors == 1:
                has_a_i_3_mod4 = any(x % 4 == 3 for x in a)
                if has_a_i_3_mod4:
                    print(1)
                else:
                    print(1)
            else:
                has_a_i_3_mod4 = any(x % 4 == 3 for x in a)
                if has_a_i_3_mod4:
                    print(1)
                else:
                    print(2)
        elif k == 5:
            has_div5 = any(x % 5 == 0 for x in a)
            if has_div5:
                print(0)
            else:
                ops = [5 - x % 5 for x in a if x % 5 != 0]
                print(min(ops))

if __name__ == "__main__":
    main()