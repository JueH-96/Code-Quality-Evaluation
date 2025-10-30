import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    m = int(input[ptr])
    ptr += 1
    d = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr + n]))
    ptr += n
    B = list(map(int, input[ptr:ptr + m]))
    ptr += m
    
    B.sort()
    
    max_sum = -1
    
    for a in A:
        x = a + d
        idx = bisect.bisect_right(B, x)
        if idx > 0:
            b_candidate = B[idx - 1]
            if b_candidate >= (a - d):
                current = a + b_candidate
                if current > max_sum:
                    max_sum = current
    
    print(max_sum if max_sum != -1 else -1)

if __name__ == "__main__":
    main()