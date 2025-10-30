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
        
        if K > 200:
            results.append("0")
            continue
            
        ans = 0
        for i in range(0, K+1):
            A = X >> i
            if A == 0:
                break
                
            if i == K:
                ans += 1
                continue
                
            if i == 0:
                power = 1 << K
                if power > N:
                    pass
                else:
                    max_X = N // power
                    if X > max_X:
                        pass
                    else:
                        low = X * power
                        high = low + power - 1
                        if high > N:
                            ans += (N - low + 1)
                        else:
                            ans += power
            else:
                down_steps = K - i
                prev_node = X >> (i-1)
                if prev_node == 2 * A:
                    other_child = 2 * A + 1
                else:
                    other_child = 2 * A
                    
                if down_steps == 1:
                    if other_child <= N:
                        ans += 1
                else:
                    power2 = 1 << (down_steps - 1)
                    if power2 > N:
                        pass
                    else:
                        max_child = N // power2
                        if other_child > max_child:
                            pass
                        else:
                            low2 = other_child * power2
                            high2 = low2 + power2 - 1
                            if high2 > N:
                                ans += (N - low2 + 1)
                            else:
                                ans += power2
                                
        results.append(str(ans))
        
    print("
".join(results))

if __name__ == "__main__":
    main()