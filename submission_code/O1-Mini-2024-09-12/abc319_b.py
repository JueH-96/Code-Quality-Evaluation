# YOUR CODE HERE
def solve():
    import sys
    N = int(sys.stdin.read())
    j_list = sorted([j for j in range(1,10) if N % j ==0])
    s = []
    for i in range(N+1):
        for j in j_list:
            k = N // j
            if k ==0:
                continue
            if i % k ==0:
                s.append(str(j))
                break
        else:
            s.append('-')
    print(''.join(s))