import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    S = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    total = sum(A)
    
    if S == 0:
        print("No")
        return
        
    if S >= total:
        remainder = S % total
        if remainder == 0:
            print("Yes")
            return
        else:
            X = remainder
    else:
        X = S
        
    B = A + A
    l = 0
    cur = 0
    found = False
    n2 = len(B)
    for r in range(n2):
        cur += B[r]
        while l <= r and (cur > X or (r - l + 1) > n):
            cur -= B[l]
            l += 1
        if cur == X:
            found = True
            break
            
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()