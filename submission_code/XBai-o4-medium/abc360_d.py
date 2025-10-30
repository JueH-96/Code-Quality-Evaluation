import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    T = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1
    X = list(map(int, input[ptr:ptr + n]))
    ptr += n
    
    A = []
    B = []
    for i in range(n):
        if S[i] == '1':
            A.append(X[i])
        else:
            B.append(X[i])
    
    A.sort()
    B.sort()
    
    ans = 0
    for a in A:
        max_pos = a + 2 * T
        start = bisect.bisect_right(B, a)
        end = bisect.bisect_right(B, max_pos)
        ans += (end - start)
    
    print(ans)

if __name__ == "__main__":
    main()