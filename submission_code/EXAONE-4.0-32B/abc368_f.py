import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    MAX_VAL = 100000
    divisors_list = [[] for _ in range(MAX_VAL+1)]
    for i in range(1, MAX_VAL+1):
        j = i
        while j <= MAX_VAL:
            divisors_list[j].append(i)
            j += i
            
    grundy = [0] * (MAX_VAL+1)
    for num in range(2, MAX_VAL+1):
        s = set()
        for d in divisors_list[num]:
            if d == num:
                continue
            s.add(grundy[d])
        mex_val = 0
        while mex_val in s:
            mex_val += 1
        grundy[num] = mex_val
        
    xor_val = 0
    for a in A:
        xor_val ^= grundy[a]
        
    print("Anna" if xor_val != 0 else "Bruno")

if __name__ == '__main__':
    main()