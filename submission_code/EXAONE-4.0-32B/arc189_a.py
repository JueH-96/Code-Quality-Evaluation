MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if A[0] != 1 or A[-1] != (n % 2):
        print(0)
        return
        
    runs_of_ones = 0
    in_run = False
    for val in A:
        if val == 1:
            if not in_run:
                in_run = True
                runs_of_ones += 1
        else:
            in_run = False
            
    if runs_of_ones == 0:
        print(1)
    else:
        ans = pow(3, runs_of_ones - 1, MOD)
        print(ans)

if __name__ == '__main__':
    main()