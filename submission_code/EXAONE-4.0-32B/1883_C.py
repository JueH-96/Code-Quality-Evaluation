import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        k = int(data[index+1])
        index += 2
        arr = list(map(int, data[index:index+n]))
        index += n
        
        if k == 2:
            found_even = False
            for x in arr:
                if x % 2 == 0:
                    found_even = True
                    break
            results.append("0" if found_even else "1")
            
        elif k == 3:
            min_ops = 10**9
            for x in arr:
                r = x % 3
                d = 0 if r == 0 else 3 - r
                if d < min_ops:
                    min_ops = d
            results.append(str(min_ops))
            
        elif k == 4:
            count_even = 0
            has_four = False
            for x in arr:
                if x % 2 == 0:
                    count_even += 1
                if x % 4 == 0:
                    has_four = True
                    
            if has_four or count_even >= 2:
                results.append("0")
            else:
                costA = 10**9
                for x in arr:
                    r = x % 4
                    d = (4 - r) % 4
                    if d < costA:
                        costA = d
                costB = 2 if count_even == 0 else 1
                results.append(str(min(costA, costB)))
                
        elif k == 5:
            min_ops = 10**9
            for x in arr:
                r = x % 5
                d = 0 if r == 0 else 5 - r
                if d < min_ops:
                    min_ops = d
            results.append(str(min_ops))
            
    print("
".join(results))

if __name__ == "__main__":
    main()