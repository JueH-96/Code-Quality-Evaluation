import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    if A == B:
        print(0)
        return
    
    total = 0
    for i in range(N):
        a = A[i]
        b = B[i]
        d_plus = (b - a) % M
        d_minus = (a - b) % M
        
        increment_valid = False
        if d_plus == 0:
            increment_valid = True
        else:
            if d_plus == 1:
                increment_valid = True
            else:
                if b > a:
                    start = a + 1
                    end = b - 1
                    if i > 0:
                        left = A[i-1]
                        if start <= left <= end:
                            increment_valid = False
                    if i < N-1:
                        right = A[i+1]
                        if start <= right <= end:
                            increment_valid = False
                else:
                    part1_start = a + 1
                    part1_end = M - 1
                    part2_start = 0
                    part2_end = b - 1
                    if i > 0:
                        left = A[i-1]
                        if (part1_start <= left <= part1_end) or (part2_start <= left <= part2_end):
                            increment_valid = False
                    if i < N-1:
                        right = A[i+1]
                        if (part1_start <= right <= part1_end) or (part2_start <= right <= part2_end):
                            increment_valid = False
        
        decrement_valid = False
        if d_minus == 0:
            decrement_valid = True
        else:
            if d_minus == 1:
                decrement_valid = True
            else:
                if a > b:
                    start = b + 1
                    end = a - 1
                    if i > 0:
                        left = A[i-1]
                        if start <= left <= end:
                            decrement_valid = False
                    if i < N-1:
                        right = A[i+1]
                        if start <= right <= end:
                            decrement_valid = False
                else:
                    part1_start = b + 1
                    part1_end = M - 1
                    part2_start = 0
                    part2_end = a - 1
                    if i > 0:
                        left = A[i-1]
                        if (part1_start <= left <= part1_end) or (part2_start <= left <= part2_end):
                            decrement_valid = False
                    if i < N-1:
                        right = A[i+1]
                        if (part1_start <= right <= part1_end) or (part2_start <= right <= part2_end):
                            decrement_valid = False
        
        if not increment_valid and not decrement_valid:
            print(-1)
            return
        
        if increment_valid and decrement_valid:
            total += min(d_plus, d_minus)
        elif increment_valid:
            total += d_plus
        else:
            total += d_minus
    
    print(total)

if __name__ == "__main__":
    main()