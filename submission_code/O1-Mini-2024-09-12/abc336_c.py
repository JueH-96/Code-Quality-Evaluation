def solve():
    import sys
    N_str = sys.stdin.read().strip()
    N = int(N_str)
    if N == 1:
        print(0)
        return
    N -=1
    digits = []
    while N >0:
        digits.append(N %5)
        N = N//5
    mapping = {0:'0',1:'2',2:'4',3:'6',4:'8'}
    mapped_digits = ''.join(mapping[d] for d in reversed(digits))
    print(mapped_digits)