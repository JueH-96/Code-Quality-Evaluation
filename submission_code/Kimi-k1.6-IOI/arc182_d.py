import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    if A == B:
        print(0)
        return
    
    # Check if any B[i] == A[i+1] for i < N-1
    possible = True
    for i in range(N-1):
        if B[i] == A[i+1]:
            possible = False
            break
    if not possible:
        print(-1)
        return
    
    total = 0
    for i in range(N):
        a = A[i]
        b = B[i]
        if a == b:
            continue
        
        forbidden = set()
        if i > 0:
            forbidden.add(B[i-1])
        if i < N-1:
            forbidden.add(A[i+1])
        
        valid_inc = True
        valid_dec = True
        steps_inc = (b - a) % M
        steps_dec = (a - b) % M
        
        for f in forbidden:
            # Check increasing path
            if (a <= b and a <= f <= b) or (a > b and (f >= a or f <= b)):
                valid_inc = False
            # Check decreasing path
            if (b <= a and b <= f <= a) or (b > a and (f >= b or f <= a)):
                valid_dec = False
        
        options = []
        if valid_inc:
            options.append(steps_inc)
        if valid_dec:
            options.append(steps_dec)
        
        if not options:
            print(-1)
            return
        
        total += min(options)
    
    print(total)

if __name__ == '__main__':
    main()