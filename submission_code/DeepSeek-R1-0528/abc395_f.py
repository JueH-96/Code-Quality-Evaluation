import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    x = int(data[1])
    teeth = []
    total = 0
    min_sum = 10**18
    
    index = 2
    for i in range(n):
        u = int(data[index])
        d = int(data[index+1])
        index += 2
        teeth.append((u, d))
        total += u + d
        if u + d < min_sum:
            min_sum = u + d
            
    min_consec = 10**18
    for i in range(n-1):
        u1, d1 = teeth[i]
        u2, d2 = teeth[i+1]
        temp = min(u1 + d2, u2 + d1)
        candidate = temp + x
        if candidate < min_consec:
            min_consec = candidate
            
    H0 = min(min_sum, min_consec)
    ans = total - n * H0
    print(ans)

if __name__ == "__main__":
    main()