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
        if k == 2:
            has_even = any(x % 2 == 0 for x in a)
            print(0 if has_even else 1)
        elif k == 3:
            has_three = any(x % 3 == 0 for x in a)
            if has_three:
                print(0)
            else:
                min_steps = min((3 - (x % 3)) % 3 for x in a)
                print(min_steps)
        elif k == 5:
            has_five = any(x % 5 == 0 for x in a)
            if has_five:
                print(0)
            else:
                min_steps = min((5 - (x % 5)) % 5 for x in a)
                print(min_steps)
        else: # k == 4
            count_div4 = sum(1 for x in a if x % 4 == 0)
            count_even = sum(1 for x in a if x % 2 == 0)
            if count_div4 > 0 or count_even >= 2:
                print(0)
            else:
                option1 = min((4 - (x % 4)) % 4 for x in a)
                if count_even == 0:
                    option2 = 2
                else:
                    option2 = 1
                print(min(option1, option2))
                
if __name__ == "__main__":
    main()