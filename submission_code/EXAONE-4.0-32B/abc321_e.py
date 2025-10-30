import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        N = int(data[index])
        X = int(data[index+1])
        K = int(data[index+2])
        index += 3
        
        ans = 0
        for j in range(0, K+1):
            A = X >> j
            if A == 0:
                break
                
            if j == K:
                ans += 1
            else:
                if j == 0:
                    if K <= 1000:
                        low_val = X << K
                        if low_val <= N:
                            high_val = low_val + (1 << K) - 1
                            if high_val > N:
                                high_val = N
                            ans += high_val - low_val + 1
                else:
                    prev = X >> (j-1)
                    c1 = A * 2
                    c2 = c1 + 1
                    if prev == c1:
                        other = c2
                    else:
                        other = c1
                    
                    if other > N:
                        continue
                        
                    d = K - j - 1
                    if d < 0:
                        continue
                    if d > 1000:
                        continue
                    
                    low_val = other << d
                    if low_val > N:
                        continue
                    high_val = low_val + (1 << d) - 1
                    if high_val > N:
                        high_val = N
                    ans += high_val - low_val + 1
                    
        results.append(str(ans))
    
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()