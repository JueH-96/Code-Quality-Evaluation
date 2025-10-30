# YOUR CODE HERE
import sys

def main():
    import sys
    import math
    from sys import stdin
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx]); idx +=1
    results = []
    for _ in range(t):
        n, k = int(input[idx]), int(input[idx+1]); idx +=2
        a = list(map(int, input[idx:idx+n])); idx +=n
        if k ==2:
            has_div = any(x %2 ==0 for x in a)
            if has_div:
                results.append(0)
            else:
                results.append(1)
        elif k ==3:
            has_div = any(x %3 ==0 for x in a)
            if has_div:
                results.append(0)
            else:
                min_op = min( (3 - (x %3)) for x in a )
                results.append(min_op)
        elif k ==4:
            has_div4 = any(x %4 ==0 for x in a)
            min_op1 = min( (4 - (x %4))%4 for x in a )
            count_even = sum(1 for x in a if x%2 ==0)
            if count_even >=2:
                op2 =0
            elif count_even ==1:
                op2 =1
            else:
                op2 =2
            if has_div4:
                results.append(0)
            else:
                min_op = min(min_op1, op2)
                results.append(min_op)
        elif k ==5:
            has_div = any(x %5 ==0 for x in a)
            if has_div:
                results.append(0)
            else:
                min_op = min( (5 - (x %5)) for x in a )
                results.append(min_op)
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()