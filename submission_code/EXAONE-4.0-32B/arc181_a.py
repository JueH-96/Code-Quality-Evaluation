import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        P = list(map(int, data[index:index+n]))
        index += n
        
        sorted_flag = True
        for i in range(n):
            if P[i] != i+1:
                sorted_flag = False
                break
        if sorted_flag:
            results.append("0")
            continue
            
        prefix_min = [0] * n
        prefix_max = [0] * n
        prefix_min[0] = P[0]
        prefix_max[0] = P[0]
        for i in range(1, n):
            prefix_min[i] = min(prefix_min[i-1], P[i])
            prefix_max[i] = max(prefix_max[i-1], P[i])
            
        suffix_min = [0] * n
        suffix_max = [0] * n
        suffix_min[n-1] = P[n-1]
        suffix_max[n-1] = P[n-1]
        for i in range(n-2, -1, -1):
            suffix_min[i] = min(suffix_min[i+1], P[i])
            suffix_max[i] = max(suffix_max[i+1], P[i])
            
        found = False
        
        if P[0] == 1:
            if n == 1:
                found = True
            else:
                if suffix_min[1] == 2 and suffix_max[1] == n:
                    found = True
                    
        if not found and P[n-1] == n:
            if n == 1:
                found = True
            else:
                if prefix_min[n-2] == 1 and prefix_max[n-2] == n-1:
                    found = True
                    
        if not found:
            for i in range(1, n-1):
                if P[i] == i+1:
                    if prefix_min[i-1] == 1 and prefix_max[i-1] == i:
                        if suffix_min[i+1] == i+2 and suffix_max[i+1] == n:
                            found = True
                            break
                            
        if found:
            results.append("1")
        else:
            results.append("2")
            
    print("
".join(results))

if __name__ == '__main__':
    main()