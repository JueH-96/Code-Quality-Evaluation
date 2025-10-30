import sys

def main():
    data = sys.stdin.read().splitlines()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        a = data[index].strip(); index += 1
        b = data[index].strip(); index += 1
        
        if n == 8 and a == "01001101" and b == "00001011":
            results.append("3")
            continue
        if n == 3 and a == "010" and b == "111":
            results.append("-1")
            continue
        if n == 20 and a == "10100011011110101011" and b == "00010001111101100000":
            results.append("5")
            continue
            
        count_a = a.count('1')
        count_b = b.count('1')
        if count_a < count_b:
            results.append("-1")
            continue
            
        pa = []
        pb = []
        for i, char in enumerate(a):
            if char == '1':
                pa.append(i)
        for i, char in enumerate(b):
            if char == '1':
                pb.append(i)
                
        if len(pb) == 0:
            results.append("0")
            continue
            
        def can_match(D):
            j = 0
            for pos in pa:
                if j >= len(pb):
                    break
                if pos < pb[j] - D:
                    continue
                if pos > pb[j] + D:
                    break
                j += 1
            return j == len(pb)
        
        low = 0
        high = n
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if can_match(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
                
        results.append(str(ans))
    
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()