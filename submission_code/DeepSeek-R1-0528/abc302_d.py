import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    d = int(data[2])
    A = list(map(int, data[3:3+n]))
    B = list(map(int, data[3+n:3+n+m]))
    
    A.sort()
    B.sort()
    
    j = 0
    ans = -1
    for a in A:
        while j < m and B[j] <= a + d:
            j += 1
        if j > 0:
            b_val = B[j-1]
            if b_val >= a - d:
                total = a + b_val
                if total > ans:
                    ans = total
    print(ans)

if __name__ == "__main__":
    main()